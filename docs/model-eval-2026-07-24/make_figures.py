"""Regenerate the curated figures for the 2026-07-24 eval report.

Reads the scored grid runs (outputs/runs/*__L<lvl>__pilot for sweep grid-full) and
writes the report's figures into ./figures/. Curated set (researcher-value first):
  1 accuracy_by_mode      — accuracy vs difficulty, content|symbolic (the core result)
  2 modality_gap          — symbolic-content vs difficulty (representation sensitivity)
  3 scaling_<family>      — score vs model size per level, content|symbolic (scaling law)
  4 censored_small_models — raw vs censored: reasoning vs length-control for 9B/4B
  5 task_profile_<model>  — per-task x level for ONE reference model (apples-to-apples)

Combined content/symbolic "overall" scores are never produced.
"""
from __future__ import annotations
import glob, json, os, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
SWEEP = "grid-full"
LEVELS = [1, 3, 5, 10, 15]
ROSTER = ["qwen3.6-27b", "qwen3.5-27b", "gemma-4-31b-it", "qwen3.5-9b", "qwen3.5-4b",
          "gemma-4-E4B-it", "llama-3.1-8b-instruct"]
COLORS = {"qwen3.6-27b": "#0072B2", "qwen3.5-27b": "#D55E00", "gemma-4-31b-it": "#009E73",
          "qwen3.5-9b": "#CC79A7", "qwen3.5-4b": "#E69F00", "gemma-4-E4B-it": "#56B4E9",
          "llama-3.1-8b-instruct": "#666666"}
MARKERS = {"qwen3.6-27b": "o", "qwen3.5-27b": "s", "gemma-4-31b-it": "^",
           "qwen3.5-9b": "D", "qwen3.5-4b": "v", "gemma-4-E4B-it": "P",
           "llama-3.1-8b-instruct": "X"}
FAMILY_SIZE = {"qwen3.5-27b": ("qwen3.5", 27), "qwen3.5-9b": ("qwen3.5", 9),
               "qwen3.5-4b": ("qwen3.5", 4), "gemma-4-31b-it": ("gemma-4", 31),
               "gemma-4-E4B-it": ("gemma-4", 4)}
RUN_RE = re.compile(r"__(?P<model>.+?)__L(?P<lvl>\d{2})__pilot$")


def load():
    """{model:{level:{'content':x,'symbolic':x,'trunc':x,'raw':x,'censored':x}}}"""
    data = {}
    for d in sorted(glob.glob(str(ROOT / "outputs/runs/*__L*__pilot"))):
        m = RUN_RE.search(d)
        if not m:
            continue
        met = _rj(Path(d) / "metrics.json"); run = _rj(Path(d) / "run.json")
        if not met or not run or run.get("sweep_id") != SWEEP:
            continue
        model, lv = m.group("model"), int(m.group("lvl"))
        rec = {"content": met["by_mode"]["content"]["mean_score"],
               "symbolic": met["by_mode"]["symbolic"]["mean_score"],
               "trunc": met["overall"]["truncated_rate"],
               "by_task_mode": met.get("by_task_mode", {})}
        rows = [json.loads(x) for x in open(Path(d) / "samples.jsonl")] \
            if (Path(d) / "samples.jsonl").exists() else []
        if rows:
            raw = [r["score"] for r in rows]
            cens = [r["score"] for r in rows if not r.get("truncated")]
            rec["raw"] = sum(raw) / len(raw)
            rec["censored"] = (sum(cens) / len(cens)) if cens else None
        data.setdefault(model, {})[lv] = rec
    return data


def _rj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _models(data):
    return [m for m in ROSTER if m in data]


def _axline(ax):
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels([f"L{l}" for l in LEVELS])
    ax.set_xlabel("difficulty level")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)


def fig1_accuracy_by_mode(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, mode in zip(axes, ("content", "symbolic")):
        for m in _models(data):
            xs = [i for i, l in enumerate(LEVELS) if l in data[m]]
            ys = [data[m][LEVELS[i]][mode] for i in xs]
            ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], lw=2, ms=7, mec="white",
                    mew=1.2, label=m)
            if xs:
                ax.annotate(m, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=COLORS[m])
        _axline(ax); ax.set_ylim(0, 1); ax.set_title(mode, loc="left", fontweight="bold")
    axes[0].set_ylabel("mean score (0–1)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("Accuracy vs difficulty, by answer mode", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig1_accuracy_by_mode.png", bbox_inches="tight")
    plt.close(fig)


def fig2_modality_gap(data):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    for m in _models(data):
        xs = [i for i, l in enumerate(LEVELS) if l in data[m]]
        ys = [data[m][LEVELS[i]]["symbolic"] - data[m][LEVELS[i]]["content"] for i in xs]
        ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], lw=2, ms=7, mec="white",
                mew=1.2, label=m)
        if xs:
            ax.annotate(m, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=COLORS[m])
    ax.axhline(0, color="#888", lw=1, ls="--")
    _axline(ax); ax.set_ylabel("symbolic − content  (gap)")
    ax.set_title("Modality gap vs difficulty  (>0 = symbolic easier)", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="best", ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig2_modality_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig3_scaling(data):
    fam = {}
    for m in _models(data):
        if m in FAMILY_SIZE:
            f, s = FAMILY_SIZE[m]; fam.setdefault(f, []).append((s, m))
    for f, members in fam.items():
        members = sorted(members)
        if len(members) < 2:
            continue
        sizes = [s for s, _ in members]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
        cmap = plt.get_cmap("viridis")
        for ax, mode in zip(axes, ("content", "symbolic")):
            for li, lv in enumerate(LEVELS):
                xs = [s for s, m in members if lv in data[m]]
                ys = [data[m][lv][mode] for s, m in members if lv in data[m]]
                if len(xs) >= 2:
                    ax.plot(xs, ys, marker="o", lw=2, ms=6, color=cmap(li / (len(LEVELS) - 1)),
                            label=f"L{lv}")
            ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{s}B" for s in sizes])
            ax.set_xlabel("model size (params, log)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
            ax.set_ylim(0, 1); ax.set_title(mode, loc="left", fontweight="bold")
        axes[0].set_ylabel("mean score (0–1)")
        axes[1].legend(frameon=False, fontsize=8, title="level", loc="lower right")
        fig.suptitle(f"Size scaling — {f}", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT / f"fig3_scaling_{f}.png", bbox_inches="tight")
        plt.close(fig)


def fig4_censored(data):
    small = [m for m in ("qwen3.5-9b", "qwen3.5-4b") if m in data]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    for m in small:
        xs = [i for i, l in enumerate(LEVELS) if l in data[m] and data[m][l].get("censored") is not None]
        raw = [data[m][LEVELS[i]]["raw"] for i in xs]
        cen = [data[m][LEVELS[i]]["censored"] for i in xs]
        ax.plot(xs, cen, color=COLORS[m], marker=MARKERS[m], lw=2, ms=7, mec="white", mew=1.2,
                label=f"{m} — censored")
        ax.plot(xs, raw, color=COLORS[m], marker=MARKERS[m], lw=1.5, ms=6, ls="--", alpha=0.75,
                label=f"{m} — raw")
        for i, r, c in zip(xs, raw, cen):
            ax.annotate("", (i, c));
    _axline(ax); ax.set_ylim(0, 1); ax.set_ylabel("mean score (0–1)")
    ax.set_title("Small models: raw vs censored (truncated items dropped)", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, -0.02, "Solid = censored (reasoning on completed answers); "
             "dashed = raw. The wide gap is length-control (truncation), not reasoning.",
             fontsize=7.5, color="#555")
    fig.tight_layout(); fig.savefig(OUT / "fig4_censored_small_models.png", bbox_inches="tight")
    plt.close(fig)


def fig5_task_profile(data, model="qwen3.6-27b"):
    if model not in data:
        return
    tasks = sorted({k.split("|")[0] for lv in data[model].values() for k in lv["by_task_mode"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=150, sharey=True)
    im = None
    for ax, mode in zip(axes, ("content", "symbolic")):
        grid = np.full((len(tasks), len(LEVELS)), np.nan)
        for ti, t in enumerate(tasks):
            for li, lv in enumerate(LEVELS):
                v = data[model].get(lv, {}).get("by_task_mode", {}).get(f"{t}|{mode}")
                if v:
                    grid[ti, li] = v["mean_score"]
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels([f"L{l}" for l in LEVELS])
        ax.set_title(mode, loc="left", fontweight="bold")
        for ti in range(len(tasks)):
            for li in range(len(LEVELS)):
                v = grid[ti, li]
                if v == v:
                    ax.text(li, ti, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v < 0.6 else "black")
    axes[0].set_yticks(range(len(tasks))); axes[0].set_yticklabels(tasks, fontsize=8)
    fig.suptitle(f"Task difficulty — {model} (single model; not cross-family)", x=0.01,
                 ha="left", fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="mean score")
    fig.savefig(OUT / f"fig5_task_profile_{model}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    fig1_accuracy_by_mode(data)
    fig2_modality_gap(data)
    fig3_scaling(data)
    fig4_censored(data)
    fig5_task_profile(data)
    print("models:", _models(data))
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))
