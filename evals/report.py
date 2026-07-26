"""Cross-model tables from a set of scored run directories."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import artifacts  # noqa: E402


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
        runs.append({"dir": str(d), "run": run, "metrics": metrics,
                     "model": (run.get("model") or {}).get("name", d.name)})
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


def _mode_score(run: dict, mode: str):
    return (run["metrics"].get("by_mode", {}).get(mode) or {}).get("mean_score")


def build_report(runs: List[dict]) -> str:
    # Ordered by symbolic score, stated as such. There is no single ranking:
    # ordering by a content+symbolic mean would rank models on a quantity that
    # does not exist, and the two orders genuinely differ.
    runs = sorted(runs, key=lambda r: -(_mode_score(r, "symbolic") or 0))
    L = []
    L.append("# ArgGYM pilot evaluation — cross-model report\n")
    L.append(f"Generated {datetime.now().isoformat(timespec='seconds')}\n")

    meta = runs[0]["metrics"].get("_meta", {}) if runs else {}
    L.append(f"Taskset: `{meta.get('taskset_id')}`  \n"
             f"KB sha256: `{(meta.get('kb_sha256') or '')[:16]}`\n")

    L.append("\n## Scores by mode\n")
    L.append("Content and symbolic are separate columns and are never averaged: "
             "they are different representations of the same tasks, so their mean "
             "is not a quantity. Rows are ordered by symbolic score.\n")
    rows = []
    for r in runs:
        o = r["metrics"]["overall"]
        con = r["metrics"].get("by_mode", {}).get("content") or {}
        sym = r["metrics"].get("by_mode", {}).get("symbolic") or {}
        gap = (None if not (con.get("mean_score") is not None
                            and sym.get("mean_score") is not None)
               else sym["mean_score"] - con["mean_score"])
        rows.append([r["model"], o.get("n"),
                     _fmt(con.get("mean_score"), 4), _fmt(con.get("stderr"), 4),
                     _fmt(sym.get("mean_score"), 4), _fmt(sym.get("stderr"), 4),
                     _fmt(gap, 4),
                     _fmt(o.get("no_answer_region_rate")),
                     _fmt(o.get("truncated_rate")),
                     _fmt(o.get("api_error_rate")),
                     _fmt(o.get("mean_completion_tokens"), 0)])
    L.append(table(rows, ["model", "n", "content", "±", "symbolic", "±",
                          "gap (sym−con)", "no_region", "trunc", "api_err",
                          "mean_tok"]))

    # Both breakdowns are keyed <thing>|<mode>, so every column is one mode. The
    # cross-mode by_level / by_task groups carry no scores by construction.
    for group, title in (("by_level_mode", "By level"),
                         ("by_task_mode", "By task")):
        for mode in ("content", "symbolic"):
            L.append(f"\n\n## {title} — {mode}\n")
            keys = [k for k in sorted({k for r in runs
                                       for k in r["metrics"].get(group, {})})
                    if k.endswith("|" + mode)]
            rows = []
            for r in runs:
                g = r["metrics"].get(group, {})
                rows.append([r["model"]]
                            + [_fmt(g.get(k, {}).get("mean_score")) for k in keys])
            L.append(table(rows, ["model"] + [k.rsplit("|", 1)[0] for k in keys]))

    L.append("\n\n## Audit signals — tasks with high well-formed-but-zero rates\n")
    L.append("High `zero_score_with_valid_region` means a parseable answer still "
             "scored 0: either genuinely wrong, or a parser that cannot read a "
             "valid answer. Read a sample of the raw generations for any task that "
             "stands out before trusting its score.\n")
    rows = []
    keys = sorted({k for r in runs for k in r["metrics"].get("by_task", {})})
    for k in keys:
        vals = [r["metrics"]["by_task"].get(k, {}) for r in runs]
        zr = [v.get("zero_score_with_valid_region_rate") for v in vals if v]
        nr = [v.get("no_answer_region_rate") for v in vals if v]
        ms = [v.get("mean_score") for v in vals if v]
        if not zr:
            continue
        rows.append([k, _fmt(sum(x or 0 for x in zr) / len(zr)),
                     _fmt(sum(x or 0 for x in nr) / len(nr)),
                     _fmt(sum(x or 0 for x in ms) / len(ms))])
    rows.sort(key=lambda r: -float(r[1]))
    L.append(table(rows, ["task", "mean zero_valid_region", "mean no_region", "mean score"]))
    return "\n".join(L)


def write_csv(path: Path, runs: List[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "group", "key", "n", "mean_score", "stderr",
                    "perfect_rate", "no_answer_region_rate", "truncated_rate",
                    "api_error_rate", "zero_score_with_valid_region_rate"])
        for r in runs:
            m = r["metrics"]
            for group in ("overall", "by_task", "by_mode", "by_level", "by_task_mode"):
                items = {"all": m["overall"]}.items() if group == "overall" \
                    else m.get(group, {}).items()
                for key, v in items:
                    w.writerow([r["model"], group, key, v.get("n"),
                                v.get("mean_score"), v.get("stderr"),
                                v.get("perfect_rate"), v.get("no_answer_region_rate"),
                                v.get("truncated_rate"), v.get("api_error_rate"),
                                v.get("zero_score_with_valid_region_rate")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="outputs/runs/*")
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = load_runs(a.runs_glob, a.sweep)
    if not runs:
        print("no scored runs found")
        return

    out = Path(a.out) if a.out else Path("outputs/reports") / \
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
