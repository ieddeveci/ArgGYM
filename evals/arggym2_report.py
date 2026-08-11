"""Cross-model tables from a set of scored ArgGYM_v2 run directories.

ArgGYM_v2 is reported on its own. It shares no task, scorer or answer
representation with the v1 generator, so no table here may carry a v1 number
beside a v2 one: they are different benchmarks, and a reader who sees them
adjacent will compare them.

Named `arggym2_` rather than `v2_` because `evals/v2_report.py` already means
something else in this repo -- the paired-content experiment on the v1
benchmark. Two unrelated things called v2 is confusing enough without the
filenames colliding too.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import artifacts  # noqa: E402

# Tasks answered with directives, where efficiency against the verified minimum
# is defined. The rest are prediction/extraction tasks with no moves to make.
DIRECTIVE_TASKS = ("attack", "defence", "attack_defense",
                   "counter_argument", "counter_argument_strict",
                   "preference_construction")


def load_runs(pattern: str, sweep: str | None = None) -> List[dict]:
    runs = []
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        metrics = artifacts.read_json(d / "metrics.json")
        run = artifacts.read_json(d / "run.json")
        if not metrics or not run:
            continue
        if sweep and run.get("sweep_id") != sweep:
            continue
        if (metrics.get("_meta") or {}).get("benchmark") != "ArgGYM_v2":
            # A v1 run dir caught by the glob. Skipped rather than rendered, so
            # a v1 score can never appear inside a v2 table.
            continue
        # Each run dir is one (model, level) cell, because the grid driver runs
        # every level as a standalone eval. Label rows with both: a bare model
        # name would repeat five times with no way to tell the levels apart.
        name = (run.get("model") or {}).get("name", d.name)
        lvl = re.search(r"__L(\d+)__", d.name)
        runs.append({"dir": str(d), "run": run, "metrics": metrics,
                     "model_name": name,
                     "level": int(lvl.group(1)) if lvl else None,
                     "model": f"{name} L{int(lvl.group(1)):02d}" if lvl else name})
    runs.sort(key=lambda r: (r["model_name"], r["level"] if r["level"] else 0))
    return runs


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def _task_mean(run: dict, task: str, field: str = "mean_score"):
    return (run["metrics"].get("by_task", {}).get(task) or {}).get(field)


def _macro(run: dict, tasks, field: str = "mean_score"):
    """Unweighted mean over per-task means.

    A ranking aid, reported beside the per-task table and never instead of it:
    it summarises eleven separate scores rather than measuring one scale.
    Unweighted so a task with more items cannot silently dominate.
    """
    vals = [v for v in (_task_mean(run, t, field) for t in tasks) if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_report(runs: List[dict]) -> str:
    tasks = sorted({t for r in runs for t in r["metrics"].get("by_task", {})})
    # Ordered by (model, level), not by score. Each row is one (model, level)
    # cell, so a score ordering interleaves a model's levels with other models'
    # and makes the difficulty curve — the thing these tables are for —
    # impossible to read off.
    runs = sorted(runs, key=lambda r: (r.get("model_name", r["model"]),
                                       r.get("level") or 0))

    L = []
    L.append("# ArgGYM_v2 — cross-model report\n")
    L.append(f"Generated {datetime.now().isoformat(timespec='seconds')}\n")

    meta = runs[0]["metrics"].get("_meta", {}) if runs else {}
    L.append(f"Benchmark: **ArgGYM_v2** (symbolic only)  \n"
             f"Taskset: `{meta.get('taskset_id')}`  \n"
             f"Taskset hash: `{(meta.get('taskset_hash') or '')[:16]}`\n")

    L.append("\n> Scores are per task. ArgGYM_v2's tasks do not share a scale — "
             "`min_directives` counts moves to make for the construction tasks "
             "and theory size for `status_query` and `formalization` — so the "
             "harness emits no mean across tasks. The macro-average below is an "
             "unweighted mean of per-task means, shown as a ranking aid beside "
             "the full table, never in place of it.\n")

    L.append("\n## Macro-average and plumbing\n")
    rows = []
    for r in runs:
        o = r["metrics"]["overall"]
        rows.append([r["model"], o.get("n"),
                     _fmt(_macro(r, tasks), 4),
                     _fmt(_macro(r, DIRECTIVE_TASKS, "mean_efficiency"), 4),
                     _fmt(o.get("no_answer_region_rate")),
                     _fmt(o.get("truncated_rate")),
                     _fmt(o.get("api_error_rate")),
                     _fmt(o.get("mean_completion_tokens"), 0)])
    L.append(table(rows, ["model", "n", "macro-avg", "macro-eff", "no_region",
                          "trunc", "api_err", "mean_tok"]))

    L.append("\n\n## By task\n")
    rows = [[r["model"]] + [_fmt(_task_mean(r, t)) for t in tasks] for r in runs]
    L.append(table(rows, ["model"] + tasks))

    L.append("\n\n## Efficiency on the construction tasks\n")
    L.append("Efficiency is `minimum / moves_used`, clamped to 1.0, defined only "
             "where a verified minimum exists. It is computed **only over items "
             "the model actually solved**, so it reads as \"when right, how "
             "economical\" — a model that solves little can still score high "
             "here. Read it beside the score, never instead of it.\n")
    rows = [[r["model"]] + [_fmt(_task_mean(r, t, "mean_efficiency"))
                            for t in DIRECTIVE_TASKS] for r in runs]
    L.append(table(rows, ["model"] + list(DIRECTIVE_TASKS)))

    L.append("\n\n## Success rate on the construction tasks\n")
    L.append("Fraction of items where every goal was met, before efficiency is "
             "considered. On success `score = 0.5 + 0.5 * efficiency`, so this "
             "is the half of the score that efficiency then modulates.\n")
    rows = [[r["model"]] + [_fmt(_task_mean(r, t, "success_rate"))
                            for t in DIRECTIVE_TASKS] for r in runs]
    L.append(table(rows, ["model"] + list(DIRECTIVE_TASKS)))

    for group, title in (("by_task_level", "By task and level"),
                         ("by_task_ordering", "By task and ordering")):
        keys = sorted({k for r in runs for k in r["metrics"].get(group, {})})
        for suf in sorted({k.rsplit("|", 1)[1] for k in keys}):
            sel = [k for k in keys if k.endswith("|" + suf)]
            L.append(f"\n\n## {title} — {suf}\n")
            rows = []
            for r in runs:
                g = r["metrics"].get(group, {})
                rows.append([r["model"]]
                            + [_fmt(g.get(k, {}).get("mean_score")) for k in sel])
            L.append(table(rows, ["model"] + [k.rsplit("|", 1)[0] for k in sel]))

    L.append("\n\n## Audit signals — well-formed answers that still scored zero\n")
    L.append("A high `zero_score_with_valid_region` means a parseable answer "
             "scored 0: either a genuine miss, or a scorer that cannot read a "
             "valid answer. Read raw generations for any task that stands out "
             "before trusting its number.\n")
    rows = []
    for t in tasks:
        vals = [v for v in (r["metrics"]["by_task"].get(t, {}) for r in runs) if v]
        if not vals:
            continue

        def mean(field, vals=vals):
            return sum(v.get(field) or 0 for v in vals) / len(vals)

        rows.append([t, _fmt(mean("zero_score_with_valid_region_rate")),
                     _fmt(mean("no_answer_region_rate")), _fmt(mean("mean_score"))])
    rows.sort(key=lambda r: -float(r[1]))
    L.append(table(rows, ["task", "mean zero_valid_region", "mean no_region",
                          "mean score"]))
    return "\n".join(L)


def write_csv(path: Path, runs: List[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "group", "key", "n", "mean_score", "stderr",
                    "perfect_rate", "zero_rate", "success_rate",
                    "mean_efficiency", "mean_deadlocks",
                    "no_answer_region_rate", "truncated_rate",
                    "api_error_rate", "zero_score_with_valid_region_rate"])
        for r in runs:
            m = r["metrics"]
            for group in ("overall", "by_task", "by_level", "by_ordering",
                          "by_task_level", "by_task_ordering"):
                items = ({"all": m["overall"]}.items() if group == "overall"
                         else m.get(group, {}).items())
                for key, v in items:
                    w.writerow([r["model"], group, key, v.get("n"),
                                v.get("mean_score"), v.get("stderr"),
                                v.get("perfect_rate"), v.get("zero_rate"),
                                v.get("success_rate"), v.get("mean_efficiency"),
                                v.get("mean_deadlocks"),
                                v.get("no_answer_region_rate"),
                                v.get("truncated_rate"), v.get("api_error_rate"),
                                v.get("zero_score_with_valid_region_rate")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-glob", default="outputs/runs/*__v2")
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = load_runs(a.runs_glob, a.sweep)
    if not runs:
        print("no scored ArgGYM_v2 runs found")
        return

    out = (Path(a.out) if a.out else
           Path("outputs/reports") / datetime.now().strftime("arggym2-%Y-%m-%d_%H-%M-%S"))
    out.mkdir(parents=True, exist_ok=True)

    md = build_report(runs)
    (out / "report.md").write_text(md)
    write_csv(out / "results.csv", runs)
    with open(out / "runs.json", "w") as fh:
        json.dump([{"model": r["model"], "dir": r["dir"]} for r in runs], fh, indent=2)
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
