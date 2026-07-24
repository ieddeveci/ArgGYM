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


def _mode(met, mode):
    return (met.get("by_mode", {}).get(mode) or {}).get("mean_score") if met else None


def fig_gap(data, levels, out: Path):
    """Modality gap = symbolic - content, per model across difficulty. A meaningful
    difference (not an average): how much the input representation matters, and how
    that shifts with difficulty. >0 means symbolic is easier for that model."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    for model in _models_sorted(data):
        c, mk = style(model)
        xs, ys = [], []
        for i, lv in enumerate(levels):
            met = data[model].get(lv)
            s, cc = _mode(met, "symbolic"), _mode(met, "content")
            if s is not None and cc is not None:
                xs.append(i); ys.append(s - cc)
        if not xs:
            continue
        ax.plot(xs, ys, color=c, marker=mk, lw=2, ms=7, mec="white", mew=1.2,
                label=model, zorder=3)
        ax.annotate(model, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=c)
    ax.axhline(0, color="#888", lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_xlabel("difficulty level")
    ax.set_ylabel("symbolic − content  (gap)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Modality gap vs difficulty  (symbolic − content; >0 = symbolic easier)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="best", ncol=2)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def _gap_grid(models, data, levels):
    import numpy as np
    grid = np.full((len(models), len(levels)), np.nan)
    for i, m in enumerate(models):
        for j, lv in enumerate(levels):
            s, c = _mode(data[m].get(lv), "symbolic"), _mode(data[m].get(lv), "content")
            if s is not None and c is not None:
                grid[i, j] = s - c
    return grid


def _draw_gap(ax, grid, ylabels):
    """Diverging heatmap of a signed gap centered at 0 (RdBu, CVD-safe: red =
    symbolic easier, blue = content easier, near-white = no gap)."""
    import numpy as np
    vmax = max(float(np.nanmax(np.abs(grid))) if not np.all(np.isnan(grid)) else 0.01, 0.01)
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if v == v:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.62 * vmax else "black")
    return im


def fig_score_matrix(data, levels, out_dir) -> List[str]:
    """model x level score matrix, one per mode (content, symbolic). Viridis."""
    import numpy as np
    models = _models_sorted(data)
    written = []
    for mode in ("content", "symbolic"):
        grid = np.full((len(models), len(levels)), np.nan)
        for i, m in enumerate(models):
            for j, lv in enumerate(levels):
                v = _mode(data[m].get(lv), mode)
                if v is not None:
                    grid[i, j] = v
        fig, ax = plt.subplots(
            figsize=(1.8 + 1.0 * len(levels), 1.2 + 0.55 * len(models)), dpi=150)
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("difficulty level")
        for i in range(len(models)):
            for j in range(len(levels)):
                v = grid[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v < 0.6 else "black")
        ax.set_title(f"{mode.capitalize()} score matrix — model x difficulty",
                     loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean score")
        p = out_dir / f"score_matrix_{mode}.png"
        fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        written.append(str(p))
    return written


def fig_gap_matrix(data, levels, out: Path):
    """model x level matrix of the modality gap (symbolic - content)."""
    models = _models_sorted(data)
    grid = _gap_grid(models, data, levels)
    fig, ax = plt.subplots(figsize=(1.8 + 1.0 * len(levels), 1.2 + 0.55 * len(models)),
                           dpi=150)
    im = _draw_gap(ax, grid, models)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_xlabel("difficulty level")
    ax.set_title("Modality gap matrix  (symbolic − content; red = symbolic easier)",
                 loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="symbolic − content")
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
    """task x level heatmaps, one panel per mode (content | symbolic), scores
    averaged across models present. Modes are NOT combined. Viridis is
    perceptually uniform and CVD-safe."""
    import numpy as np
    tasks = sorted({k.split("|")[0] for lm in data.values() for met in lm.values()
                    for k in met.get("by_task_mode", {})})
    if not tasks:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=150, sharey=True)
    im = None
    for ax, mode in zip(axes, ("content", "symbolic")):
        grid = np.full((len(tasks), len(levels)), np.nan)
        for ti, t in enumerate(tasks):
            for li, lv in enumerate(levels):
                key = f"{t}|{mode}"
                vals = [met["by_task_mode"][key]["mean_score"]
                        for met in (lm.get(lv) for lm in data.values())
                        if met and key in met.get("by_task_mode", {})]
                if vals:
                    grid[ti, li] = sum(vals) / len(vals)
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_title(mode, loc="left", fontweight="bold")
        for ti in range(len(tasks)):
            for li in range(len(levels)):
                v = grid[ti, li]
                if v == v:
                    ax.text(li, ti, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v < 0.6 else "black")
    axes[0].set_yticks(range(len(tasks))); axes[0].set_yticklabels(tasks, fontsize=8)
    fig.suptitle("Mean score by task x difficulty (avg over models), by mode",
                 x=0.01, ha="left", fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="mean score")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def _scaling_families(data):
    fam: Dict[str, list] = {}
    for model in data:
        f, sz = _fam_size(model)
        fam.setdefault(f, []).append((sz, model))
    return {f: sorted(v) for f, v in fam.items() if len(v) >= 2}


def fig_scaling(data, levels, out_dir) -> List[str]:
    """Per family with >=2 sizes, one scaling matrix PER MODE (content, symbolic;
    never combined): rows = model size ascending, cols = level. Reading a column
    top-to-bottom shows the size effect at that difficulty. Returns paths."""
    import numpy as np
    written = []
    for f, members in _scaling_families(data).items():
        models = [m[1] for m in members]
        labels = [f"{m[0]}B  ({m[1]})" for m in members]
        for mode in ("content", "symbolic"):
            grid = np.full((len(models), len(levels)), np.nan)
            for i, model in enumerate(models):
                for j, lv in enumerate(levels):
                    v = _mode(data[model].get(lv), mode)
                    if v is not None:
                        grid[i, j] = v
            fig, ax = plt.subplots(
                figsize=(1.6 + 1.0 * len(levels), 1.2 + 0.7 * len(models)), dpi=150)
            im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
            ax.set_yticks(range(len(models))); ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("difficulty level"); ax.set_ylabel("model size (ascending)")
            for i in range(len(models)):
                for j in range(len(levels)):
                    v = grid[i, j]
                    if v == v:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8.5, color="white" if v < 0.6 else "black")
            ax.set_title(f"Scaling: {f} ({mode}) — score by size x difficulty",
                         loc="left", fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean score")
            p = out_dir / f"scaling_{f}_{mode}.png"
            fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            written.append(str(p))
        # gap matrix for the family: size x level, symbolic - content (diverging)
        gap = _gap_grid(models, data, levels)
        fig, ax = plt.subplots(
            figsize=(1.8 + 1.0 * len(levels), 1.2 + 0.6 * len(models)), dpi=150)
        im = _draw_gap(ax, gap, labels)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_xlabel("difficulty level")
        ax.set_title(f"Scaling: {f} (gap) — symbolic − content by size x difficulty",
                     loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="symbolic − content")
        p = out_dir / f"scaling_{f}_gap.png"
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

    L.append("Content and symbolic are different input representations and are "
             "never averaged into a single score; they are reported as parallel "
             "tracks, plus their gap (symbolic − content).\n")

    L.append("\n## Figures\n")
    for f in figs:
        L.append(f"![{Path(f).stem}]({Path(f).name})")

    for mode in ("content", "symbolic"):
        L.append(f"\n\n## {mode} mode — model x level\n")
        rows = [[m] + [_cell(data[m].get(lv), ["by_mode", mode]) for lv in levels]
                for m in models]
        L.append(_tbl(["model"] + [f"L{l}" for l in levels], rows))

    L.append("\n\n## Modality gap — symbolic − content (>0 = symbolic easier)\n")
    rows = []
    for m in models:
        r = [m]
        for lv in levels:
            s, c = _mode(data[m].get(lv), "symbolic"), _mode(data[m].get(lv), "content")
            r.append(f"{s - c:+.3f}" if (s is not None and c is not None) else "-")
        rows.append(r)
    L.append(_tbl(["model"] + [f"L{l}" for l in levels], rows))

    L.append("\n\n## Truncation rate — model x level (length-cap diagnostic, not a score)\n")
    rows = []
    for m in models:
        r = [m]
        for lv in levels:
            met = data[m].get(lv)
            tr = met["overall"].get("truncated_rate") if met else None
            r.append(f"{tr:.3f}" if isinstance(tr, (int, float)) else "-")
        rows.append(r)
    L.append(_tbl(["model"] + [f"L{l}" for l in levels], rows))

    # Scaling: per family with >=2 sizes, size x level, one table per mode.
    scaling = _scaling_families(data)
    if scaling:
        L.append("\n\n## Scaling — size x difficulty, per family (by mode)\n")
        L.append("Reading a column top-to-bottom shows the effect of model size at "
                 "that difficulty. Only families with >=2 sizes appear. Modes shown "
                 "separately (never combined).\n")
        for f, members in scaling.items():
            for mode in ("content", "symbolic"):
                L.append(f"\n**{f} — {mode}**\n")
                rows = []
                for sz, m in members:
                    r = [f"{sz}B ({m})"]
                    for lv in levels:
                        r.append(_cell(data[m].get(lv), ["by_mode", mode]))
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
    for name, fn in (("difficulty_by_mode.png", fig_by_mode),
                     ("modality_gap_vs_difficulty.png", fig_gap),
                     ("modality_gap_matrix.png", fig_gap_matrix),
                     ("truncation_vs_difficulty.png", fig_truncation),
                     ("task_difficulty_heatmap_by_mode.png", fig_task_heatmap)):
        p = out / name
        try:
            fn(data, levels, p)
            if p.exists():
                figs.append(str(p))
        except Exception as e:
            print(f"  figure {name} skipped: {type(e).__name__}: {e}")
    # Score matrices (model x level, per mode) + scaling matrices (per family).
    try:
        figs += fig_score_matrix(data, levels, out)
    except Exception as e:
        print(f"  score matrices skipped: {type(e).__name__}: {e}")
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
