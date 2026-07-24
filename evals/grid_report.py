"""Report + figures for the per-(model, level) grid sweep.

Unlike evals/report.py (one run per model), the grid produces a separate scored
run per (model, level): outputs/runs/<ts>__<model>__L<lvl>__pilot. This reads them
all and builds a level x model view plus difficulty-vs-performance figures.

Robust to partial data: plots whatever (model, level) runs exist, so it can be run
mid-sweep or at completion.

    python -m evals.grid_report --sweep grid-full \
        --out workspace/model-eval-2026-07-21/grid-report
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import artifacts  # noqa: E402

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

RUN_RE = re.compile(r"__(?P<model>.+?)__L(?P<lvl>\d{2})__pilot$")

# Canonical roster order -> stable color + marker per model, regardless of which
# subset is present. Okabe-Ito (colorblind-safe); each model also gets a distinct
# marker shape, so identity survives CVD and grayscale (secondary encoding).
ROSTER = ["qwen3.6-27b", "qwen3.5-27b", "gemma-4-31b-it",
          "qwen3.5-9b", "qwen3.5-4b", "llama-3.1-8b-instruct"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Model -> (scaling family, size in billions). A scaling matrix is drawn per
# family that has >= 2 sizes present, so size effect can be read at fixed
# difficulty. qwen3.5 gives the clean 4/9/27 ladder; qwen3.6 and gemma-4 have a
# single size in the roster (gemma-4-E4B-it would add a second Gemma point).
FAMILY_SIZE = {
    "qwen3.6-27b": ("qwen3.6", 27), "qwen3.5-27b": ("qwen3.5", 27),
    "qwen3.5-9b": ("qwen3.5", 9), "qwen3.5-4b": ("qwen3.5", 4),
    "gemma-4-31b-it": ("gemma-4", 31), "gemma-4-E4B-it": ("gemma-4", 4),
    "llama-3.1-8b-instruct": ("llama-3.1", 8),
}


def _fam_size(model):
    return FAMILY_SIZE.get(model, (model, 0))


def style(model: str):
    i = ROSTER.index(model) if model in ROSTER else (hash(model) % len(COLORS))
    return COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]


def discover(pattern: str, sweep: Optional[str]) -> Dict[str, Dict[int, dict]]:
    """{model: {level: metrics}} over scored grid runs."""
    data: Dict[str, Dict[int, dict]] = {}
    for d in sorted(glob.glob(pattern)):
        m = RUN_RE.search(d)
        if not m:
            continue
        metrics = artifacts.read_json(Path(d) / "metrics.json")
        run = artifacts.read_json(Path(d) / "run.json")
        if not metrics or not run:
            continue
        if sweep and run.get("sweep_id") != sweep:
            continue
        data.setdefault(m.group("model"), {})[int(m.group("lvl"))] = metrics
    return data


def _models_sorted(data) -> List[str]:
    return sorted(data, key=lambda x: ROSTER.index(x) if x in ROSTER else 99)


def _levels_sorted(data) -> List[int]:
    return sorted({lv for lm in data.values() for lv in lm})


# ---- figures -------------------------------------------------------------

def _line_axes(ax, levels):
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("difficulty level")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def _plot_series(ax, data, levels, value_fn):
    """One line per model of value_fn(metrics) across levels; direct end-label."""
    for model in _models_sorted(data):
        c, mk = style(model)
        xs, ys = [], []
        for i, lv in enumerate(levels):
            met = data[model].get(lv)
            v = value_fn(met) if met else None
            if v is not None:
                xs.append(i); ys.append(v)
        if not xs:
            continue
        ax.plot(xs, ys, color=c, marker=mk, lw=2, ms=7, mec="white", mew=1.2,
                label=model, zorder=3)
        ax.annotate(model, (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=c)


def fig_overall(data, levels, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    _plot_series(ax, data, levels, lambda m: m["overall"].get("mean_score"))
    _line_axes(ax, levels)
    ax.set_ylabel("mean score (0–1)")
    ax.set_title("Performance vs difficulty (overall)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_by_mode(data, levels, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, mode in zip(axes, ("content", "symbolic")):
        _plot_series(ax, data, levels,
                     lambda m, md=mode: (m.get("by_mode", {}).get(md) or {}).get("mean_score"))
        _line_axes(ax, levels)
        ax.set_title(mode, loc="left", fontweight="bold")
    axes[0].set_ylabel("mean score (0–1)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("Performance vs difficulty, by answer mode", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_truncation(data, levels, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    _plot_series(ax, data, levels, lambda m: m["overall"].get("truncated_rate"))
    _line_axes(ax, levels)
    ax.set_ylim(0, None)
    ax.set_ylabel("truncated rate")
    ax.set_title("Truncation vs difficulty (length-cap hits, not reasoning)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_task_heatmap(data, levels, out: Path):
    """task x level, mean score averaged across models present. Viridis is
    perceptually uniform and CVD-safe -- the scientific default for value maps."""
    tasks = sorted({t for lm in data.values() for met in lm.values()
                    for t in met.get("by_task", {})})
    if not tasks:
        return
    import numpy as np
    grid = np.full((len(tasks), len(levels)), np.nan)
    for ti, t in enumerate(tasks):
        for li, lv in enumerate(levels):
            vals = [met["by_task"][t]["mean_score"]
                    for met in (lm.get(lv) for lm in data.values()) if met
                    and t in met.get("by_task", {})]
            if vals:
                grid[ti, li] = sum(vals) / len(vals)
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=150)
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks, fontsize=8)
    for ti in range(len(tasks)):
        for li in range(len(levels)):
            v = grid[ti, li]
            if v == v:  # not NaN
                ax.text(li, ti, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v < 0.6 else "black")
    ax.set_title("Mean score by task x difficulty (avg over models)",
                 loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean score")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_scaling(data, levels, out_dir) -> List[str]:
    """One scaling matrix per family with >=2 sizes: rows = model (size asc),
    cols = level, cell = overall mean. Reading a column top-to-bottom shows the
    size effect at that difficulty. Returns the paths written."""
    import numpy as np
    fam: Dict[str, list] = {}
    for model in data:
        f, sz = _fam_size(model)
        fam.setdefault(f, []).append((sz, model))
    written = []
    for f, members in fam.items():
        if len(members) < 2:
            continue
        members.sort()  # by size ascending
        models = [m[1] for m in members]
        labels = [f"{m[0]}B  ({m[1]})" for m in members]
        grid = np.full((len(models), len(levels)), np.nan)
        for i, model in enumerate(models):
            for j, lv in enumerate(levels):
                met = data[model].get(lv)
                if met:
                    grid[i, j] = met["overall"].get("mean_score", np.nan)
        fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(levels), 1.2 + 0.7 * len(models)),
                               dpi=150)
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_yticks(range(len(models))); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("difficulty level")
        ax.set_ylabel("model size (ascending)")
        for i in range(len(models)):
            for j in range(len(levels)):
                v = grid[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if v < 0.6 else "black")
        ax.set_title(f"Scaling: {f} — overall score by size x difficulty",
                     loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean score")
        p = out_dir / f"scaling_{f}.png"
        fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        written.append(str(p))
    return written


# ---- markdown tables -----------------------------------------------------

def _tbl(headers, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def _cell(met, path):
    if not met:
        return "-"
    node = met
    for k in path:
        node = (node or {}).get(k, {})
    v = node.get("mean_score") if isinstance(node, dict) else None
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def build_md(data, levels, figs: List[str]) -> str:
    models = _models_sorted(data)
    L = ["# ArgGYM grid evaluation — results\n"]
    meta = next((m["_meta"] for lm in data.values() for m in lm.values()
                 if m.get("_meta")), {})
    L.append(f"Taskset `{meta.get('taskset_id')}` · "
             f"models {len(models)} · levels {levels}\n")
    done = sum(len(lm) for lm in data.values())
    L.append(f"Scored (model, level) cells: **{done}** of "
             f"{len(models) * len(levels)} (roster x levels present).\n")

    L.append("\n## Figures\n")
    for f in figs:
        L.append(f"![{Path(f).stem}]({Path(f).name})")

    L.append("\n\n## Overall mean score — model x level\n")
    rows = [[m] + [_cell(data[m].get(lv), ["overall"]) for lv in levels] for m in models]
    L.append(_tbl(["model"] + [f"L{l}" for l in levels], rows))

    for mode in ("content", "symbolic"):
        L.append(f"\n\n## {mode} mode — model x level\n")
        rows = [[m] + [_cell(data[m].get(lv), ["by_mode", mode]) for lv in levels]
                for m in models]
        L.append(_tbl(["model"] + [f"L{l}" for l in levels], rows))

    L.append("\n\n## Truncation rate — model x level\n")
    rows = []
    for m in models:
        r = [m]
        for lv in levels:
            met = data[m].get(lv)
            tr = met["overall"].get("truncated_rate") if met else None
            r.append(f"{tr:.3f}" if isinstance(tr, (int, float)) else "-")
        rows.append(r)
    L.append(_tbl(["model"] + [f"L{l}" for l in levels], rows))

    # Scaling: per family with >=2 sizes, size x level (rows = size ascending).
    fam: Dict[str, list] = {}
    for m in models:
        f, sz = _fam_size(m)
        fam.setdefault(f, []).append((sz, m))
    scaling = {f: sorted(v) for f, v in fam.items() if len(v) >= 2}
    if scaling:
        L.append("\n\n## Scaling — size x difficulty (overall), per family\n")
        L.append("Reading a column top-to-bottom shows the effect of model size at "
                 "that difficulty. Only families with >=2 sizes in the roster appear.\n")
        for f, members in scaling.items():
            L.append(f"\n**{f}**\n")
            rows = []
            for sz, m in members:
                r = [f"{sz}B ({m})"]
                for lv in levels:
                    r.append(_cell(data[m].get(lv), ["overall"]))
                rows.append(r)
            L.append(_tbl(["size"] + [f"L{l}" for l in levels], rows))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="outputs/runs/*__L*__pilot")
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--out", default="workspace/model-eval-2026-07-21/grid-report")
    a = ap.parse_args()

    data = discover(a.runs_glob, a.sweep)
    if not data:
        print("no grid runs found"); return
    levels = _levels_sorted(data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    figs = []
    for name, fn in (("difficulty_vs_overall.png", fig_overall),
                     ("difficulty_by_mode.png", fig_by_mode),
                     ("truncation_vs_difficulty.png", fig_truncation),
                     ("task_difficulty_heatmap.png", fig_task_heatmap)):
        p = out / name
        try:
            fn(data, levels, p)
            if p.exists():
                figs.append(str(p))
        except Exception as e:
            print(f"  figure {name} skipped: {type(e).__name__}: {e}")
    # Scaling matrices (one per family with >=2 sizes present).
    try:
        figs += fig_scaling(data, levels, out)
    except Exception as e:
        print(f"  scaling figures skipped: {type(e).__name__}: {e}")

    (out / "grid-results.md").write_text(build_md(data, levels, figs))
    print(f"models: {_models_sorted(data)}")
    print(f"levels: {levels}")
    print(f"figures: {[Path(f).name for f in figs]}")
    print(f"wrote {out}/grid-results.md")


if __name__ == "__main__":
    main()
