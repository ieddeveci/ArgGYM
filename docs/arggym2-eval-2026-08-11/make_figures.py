"""Regenerate the curated figures for the 2026-08-11 ArgGYM_v2 report.

Reads the scored grid runs (outputs/runs/*__L<lvl>__v2 for sweep
v2-20260805-093137) and writes into ./figures/. Curated set:

  1 contamination        — fraction of items lost to truncation/API error, by
                           level. Comes FIRST because it decides which of the
                           other numbers mean anything.
  2 macro_by_level       — macro-average vs difficulty, contaminated models
                           drawn dashed so an eye cannot mistake them for
                           measurements.
  3 recognise_vs_construct — status_query against the mean of the six
                           construction tasks: the sweep's sharpest boundary.
  4 task_profile_gemma-4-31b-it — per-task x level for the strongest
                           uncontaminated model, apples-to-apples.
  5 tokens_vs_truncation — output length against truncation rate, the
                           mechanism behind figure 1.

No figure aggregates a score across tasks without labelling it a macro-average,
and no figure puts a contaminated cell on the same visual footing as a clean one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

LEVELS = [3, 6, 9, 12, 15]
ROSTER = ["qwen3.6-27b", "qwen3.5-27b", "gemma-4-31b-it", "gemma-4-E4B-it",
          "qwen3.5-9b", "qwen3.5-4b", "llama-3.1-8b-instruct"]
COLORS = {"qwen3.6-27b": "#0072B2", "qwen3.5-27b": "#D55E00",
          "gemma-4-31b-it": "#009E73", "qwen3.5-9b": "#CC79A7",
          "qwen3.5-4b": "#E69F00", "gemma-4-E4B-it": "#56B4E9",
          "llama-3.1-8b-instruct": "#666666"}
MARKERS = {"qwen3.6-27b": "o", "qwen3.5-27b": "s", "gemma-4-31b-it": "^",
           "qwen3.5-9b": "D", "qwen3.5-4b": "v", "gemma-4-E4B-it": "P",
           "llama-3.1-8b-instruct": "X"}

# Six tasks answered with directives -- the ones that ask the model to *build*
# something rather than read off a status.
CONSTRUCTION = ["attack", "defence", "attack_defense", "counter_argument",
                "counter_argument_strict", "preference_construction"]

# Above this fraction lost to truncation + API error, a cell's score is not a
# reasoning measurement. Drawn dashed throughout so the distinction is visible
# without reading a caption.
CONTAM_LIMIT = 0.20

# Run-directory suffix identifying the sweep. A tag rather than a hardcoded
# "v2" because a later sweep on a rebuilt taskset writes `__v3` dirs, and
# globbing both at once would silently average two different tasksets into one
# figure -- the exact mistake the tag exists to prevent.
def run_re(tag: str) -> "re.Pattern":
    return re.compile(rf"__(?P<model>.+?)__L(?P<lvl>\d{{2}})__{re.escape(tag)}$")


def load(tag: str = "v2") -> dict:
    rx = run_re(tag)
    cells = {}
    for d in glob.glob(str(ROOT / "outputs" / "runs" / f"*__{tag}")):
        m = rx.search(os.path.basename(d))
        if not m:
            continue
        p = Path(d) / "metrics.json"
        if not p.exists():
            continue
        cells[(m.group("model"), int(m.group("lvl")))] = json.loads(p.read_text())
    return cells


def macro(metrics: dict) -> float | None:
    vals = [v["mean_score"] for v in metrics["by_task"].values()
            if v.get("mean_score") is not None]
    return sum(vals) / len(vals) if vals else None


def lost(metrics: dict) -> float:
    o = metrics["overall"]
    return o.get("truncated_rate", 0) + o.get("api_error_rate", 0)


def series(cells, model, fn):
    xs, ys = [], []
    for lv in LEVELS:
        c = cells.get((model, lv))
        if c is None:
            continue
        v = fn(c)
        if v is not None:
            xs.append(lv)
            ys.append(v)
    return xs, ys


def _style(model, cells):
    """Dashed + hollow when the model's cells are mostly contamination.

    Averaged over the model's levels rather than judged per point: a line whose
    dash pattern changes mid-curve is harder to read than one that doesn't, and
    every model here is either clean at all levels or dirty at all levels except
    qwen3.6-27b, whose L3 (16.1%) sits just under the line. Erring toward
    "dashed" there is the safe direction -- it understates trust, and figure 1
    carries the per-level truth.
    """
    ls = [lost(c) for (m, _), c in cells.items() if m == model]
    dirty = ls and (sum(ls) / len(ls)) > CONTAM_LIMIT
    return {"linestyle": "--" if dirty else "-",
            "alpha": 0.55 if dirty else 1.0,
            "markerfacecolor": "none" if dirty else COLORS[model]}


def fig1_contamination(cells):
    """Total loss, then the two mechanisms behind it, side by side.

    Split because they are different failures with different fixes: hitting the
    token cap is the model refusing to stop, while an API error here is the
    request exceeding the 5400s wall clock. qwen3.6-27b is contaminated almost
    entirely by the second and barely at all by the first -- a single combined
    line hides that.
    """
    panels = [("truncated or API error", lost, "total loss"),
              ("truncated", lambda c: c["overall"].get("truncated_rate", 0),
               "hit the token cap"),
              ("API error", lambda c: c["overall"].get("api_error_rate", 0),
               "timed out / errored")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, (lab, fn, sub) in zip(axes, panels):
        # Models that never leave the floor are drawn as markers only, jittered
        # horizontally. Nudging them *vertically* would be a lie -- it invents a
        # rate the model does not have -- whereas an x-jitter moves a point away
        # from a level it does not claim to sit between, and the marker still
        # reads off the y-axis correctly.
        flat = [m for m in ROSTER
                if series(cells, m, fn)[0] and max(series(cells, m, fn)[1]) < 0.01]
        for m in ROSTER:
            xs, ys = series(cells, m, fn)
            if not xs:
                continue
            if m in flat:
                j = 0.28 * (flat.index(m) - (len(flat) - 1) / 2)
                ax.plot([x + j for x in xs], [y * 100 for y in ys],
                        marker=MARKERS[m], color=COLORS[m], label=m,
                        linestyle="none", markersize=5)
            else:
                ax.plot(xs, [y * 100 for y in ys], marker=MARKERS[m],
                        color=COLORS[m], label=m, linewidth=2, markersize=5)
        ax.axhline(CONTAM_LIMIT * 100, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("level")
        ax.set_title(f"{lab}\n({sub})", fontsize=10)
        ax.set_xticks(LEVELS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of items")
    axes[0].set_ylim(-3, 100)
    axes[2].text(15.4, CONTAM_LIMIT * 100, "unusable\nabove here", fontsize=8,
                 va="center")
    axes[0].legend(fontsize=7.5, loc="center right")
    fig.suptitle("1. Contamination by level — read this before any score "
                 "(models flat at 0% shown as markers, jittered sideways)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_contamination.png", dpi=150)
    plt.close(fig)


def fig2_macro_by_level(cells):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ROSTER:
        xs, ys = series(cells, m, macro)
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                linewidth=2, **_style(m, cells))
    ax.set_xlabel("level")
    ax.set_ylabel("macro-average (mean of 11 per-task means)")
    ax.set_title("2. Score vs difficulty\n"
                 "dashed = >20% of items never reached the scorer", fontsize=11)
    ax.set_xticks(LEVELS)
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_macro_by_level.png", dpi=150)
    plt.close(fig)


def fig3_recognise_vs_construct(cells):
    """The sweep's sharpest boundary: reading a theory vs building in it."""
    def rec(c):
        return c["by_task"].get("status_query", {}).get("mean_score")

    def con(c):
        vs = [c["by_task"][t]["mean_score"] for t in CONSTRUCTION
              if t in c["by_task"] and c["by_task"][t].get("mean_score") is not None]
        return sum(vs) / len(vs) if vs else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for m in ROSTER:
        st = _style(m, cells)
        xs, ys = series(cells, m, rec)
        if xs:
            a1.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=m,
                    linewidth=2, **st)
        xs, ys = series(cells, m, con)
        if xs:
            a2.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], linewidth=2, **st)
    a1.set_title("recognise: status_query")
    a2.set_title("construct: mean of 6 construction tasks")
    for a in (a1, a2):
        a.set_xlabel("level")
        a.set_xticks(LEVELS)
        a.set_ylim(0, 1.0)
        a.grid(alpha=0.3)
    a1.set_ylabel("mean score")
    a1.legend(fontsize=7)
    fig.suptitle("3. Recognising a theory is not constructing in one "
                 "(dashed = contaminated)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_recognise_vs_construct.png", dpi=150)
    plt.close(fig)


def fig4_task_profile(cells, model="gemma-4-31b-it"):
    """One uncontaminated model, every task, every level."""
    tasks = sorted({t for (m, _), c in cells.items() if m == model
                    for t in c["by_task"]})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(tasks):
        xs, ys = series(cells, model,
                        lambda c, t=t: c["by_task"].get(t, {}).get("mean_score"))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, color=cmap(i % 20),
                    label=t, linewidth=1.6)
    ax.set_xlabel("level")
    ax.set_ylabel("mean score")
    ax.set_title(f"4. Task profile — {model} (contamination <=0.9%, so these "
                 f"are measurements)", fontsize=10)
    ax.set_xticks(LEVELS)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"fig4_task_profile_{model}.png", dpi=150)
    plt.close(fig)


def fig5_tokens_vs_truncation(cells):
    """The mechanism: models that generate near the cap truncate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ROSTER:
        pts = [(c["overall"].get("mean_completion_tokens", 0),
                c["overall"].get("truncated_rate", 0) * 100)
               for (mm, _), c in sorted(cells.items()) if mm == m]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   color=COLORS[m], marker=MARKERS[m], s=70, label=m,
                   edgecolors="black", linewidths=0.4)
    # Only the qwen/llama cap is drawn. The gemma cap (16,384) is never
    # approached -- gemma's longest cell means ~8K -- so a line there would
    # suggest a constraint that is not binding on any point in this figure.
    ax.axvline(61440, color="#555555", linestyle=":", linewidth=1)
    ax.text(60600, 45, "qwen / llama cap = 61,440", fontsize=8, rotation=90,
            va="center", ha="right", color="#555555")
    ax.set_xlabel("mean completion tokens")
    ax.set_ylabel("% truncated")
    ax.set_xlim(0, 68000)
    ax.set_ylim(-4, 100)
    ax.set_title("5. Truncation tracks output length, and output length is a\n"
                 "property of the model family — not of the task", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_tokens_vs_truncation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v2",
                    help="run-directory suffix to read (v2, v3, ...). One tag "
                         "per invocation: mixing sweeps built on different "
                         "tasksets into one figure would compare models on "
                         "different questions.")
    tag = ap.parse_args().tag
    cells = load(tag)
    print(f"loaded {len(cells)} scored cells (tag {tag})")
    if not cells:
        raise SystemExit(f"no scored runs matching outputs/runs/*__{tag}")
    fig1_contamination(cells)
    fig2_macro_by_level(cells)
    fig3_recognise_vs_construct(cells)
    fig4_task_profile(cells)
    fig5_tokens_vs_truncation(cells)
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(Path(__file__).resolve().parent))


if __name__ == "__main__":
    main()
