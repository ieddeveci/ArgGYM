"""Report for the v2 paired-content benchmark (issue #15).

Reads the scored runs of the `paired-v2` sweep and the paired taskset, and
computes the two v2 metrics per (model, level). Content and symbolic scores are
reported as parallel tracks plus their SIGNED gap -- never averaged.

  Metric 1 — representation gap. For each structure, the symbolic twin's score
    vs. its content rendering-0 score (SAME ops -> gold held constant). The mean
    gap (symbolic - content) isolates the representation effect, with the logical
    instance fixed (unlike v1, where the two modes are different instances).

  Metric 2 — surface variance. For structures with K>=2 content renderings (same
    structure, different KB records / phrasings, matched gold), how much the
    model's correctness varies across renderings: mean within-structure stdev of
    score and the "flip rate" (fraction of structures where the model is not
    all-correct-or-all-wrong across renderings). High = phrasing/world-knowledge
    drives the answer rather than the logic.

Writes v2-results.md + figures into the given output dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ROSTER = ["qwen3.6-27b", "qwen3.5-27b", "gemma-4-31b-it", "qwen3.5-9b",
          "qwen3.5-4b", "gemma-4-E4B-it", "llama-3.1-8b-instruct"]
COLORS = {"qwen3.6-27b": "#0072B2", "qwen3.5-27b": "#D55E00", "gemma-4-31b-it": "#009E73",
          "qwen3.5-9b": "#CC79A7", "qwen3.5-4b": "#E69F00", "gemma-4-E4B-it": "#56B4E9",
          "llama-3.1-8b-instruct": "#999999"}
MARKERS = {"qwen3.6-27b": "o", "qwen3.5-27b": "s", "gemma-4-31b-it": "^",
           "qwen3.5-9b": "D", "qwen3.5-4b": "v", "gemma-4-E4B-it": "P",
           "llama-3.1-8b-instruct": "X"}


def _rj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def load_taskset_meta(taskset_id):
    """sample_id -> {structure_id, rendering_id, mode, level}."""
    import gzip
    d = ROOT / "data" / "tasksets" / taskset_id
    plain, gz = d / "taskset.jsonl", d / "taskset.jsonl.gz"
    opener, path = (open, plain) if plain.exists() else (lambda p: gzip.open(p, "rt"), gz)
    meta = {}
    with opener(path) as fh:
        for line in fh:
            r = json.loads(line)
            m = r["entry"]["metadata"]
            meta[r["sample_id"]] = {"structure_id": m["structure_id"],
                                    "rendering_id": m["rendering_id"],
                                    "mode": r["mode"], "level": r["level"]}
    return meta


def load(sweep):
    """{model: {level: {sample rows joined with structure meta}}}."""
    data = defaultdict(lambda: defaultdict(list))
    ts_meta_cache = {}
    for d in sorted(glob.glob(str(ROOT / "outputs/runs/*"))):
        run = _rj(Path(d) / "run.json")
        if not run or run.get("sweep_id") != sweep:
            continue
        sf = Path(d) / "samples.jsonl"
        if not sf.exists():
            continue
        model = (run.get("model") or {}).get("name")
        ts_id = run.get("taskset_id")
        if ts_id not in ts_meta_cache:
            ts_meta_cache[ts_id] = load_taskset_meta(ts_id)
        meta = ts_meta_cache[ts_id]
        for line in open(sf):
            s = json.loads(line)
            m = meta.get(s["sample_id"])
            if not m:
                continue
            data[model][m["level"]].append({**s, **m})
    return data


def _mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def metrics(rows):
    """Compute per-(model,level) metrics from joined sample rows."""
    # index by structure
    by_struct = defaultdict(lambda: {"content": {}, "symbolic": {}})
    for r in rows:
        by_struct[r["structure_id"]][r["mode"]][r["rendering_id"]] = r["score"]

    content_r0, symbolic = [], []          # Metric 1: paired
    content_all = []                        # content accuracy (all renderings)
    stdevs, flips, n_multi = [], 0, 0       # Metric 2
    for sid, d in by_struct.items():
        c, s = d["content"], d["symbolic"]
        content_all.extend(c.values())
        if 0 in c and 0 in s:
            content_r0.append(c[0])
            symbolic.append(s[0])
        if len(c) >= 2:
            n_multi += 1
            vals = list(c.values())
            stdevs.append(statistics.pstdev(vals))
            perfect = [v >= 1.0 for v in vals]
            if any(perfect) and not all(perfect):
                flips += 1
    sym_mean = _mean([symbolic[i] for i in range(len(symbolic))])
    con_mean = _mean(content_all)
    gap = _mean([symbolic[i] - content_r0[i] for i in range(len(symbolic))]) \
        if symbolic else float("nan")
    return {
        "n_struct": len(by_struct),
        "content_mean": con_mean,
        "symbolic_mean": _mean(symbolic),
        "gap": gap,                          # symbolic - content, paired
        "n_multi": n_multi,
        "surface_stdev": _mean(stdevs),      # mean within-structure score stdev
        "flip_rate": (flips / n_multi) if n_multi else float("nan"),
    }


def build(data, levels):
    return {m: {lv: metrics(data[m][lv]) for lv in levels if data[m].get(lv)}
            for m in ROSTER if m in data}


def _models(res):
    return [m for m in ROSTER if m in res]


def fig_accuracy(res, levels, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, mode in zip(axes, ("content_mean", "symbolic_mean")):
        for m in _models(res):
            xs = [i for i, lv in enumerate(levels) if lv in res[m]]
            ys = [res[m][levels[i]][mode] for i in xs]
            ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], lw=2, ms=7,
                    mec="white", mew=1.2, label=m)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
        ax.set_ylim(0, 1); ax.set_xlabel("difficulty")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        ax.set_title(mode.split("_")[0], loc="left", fontweight="bold")
    axes[0].set_ylabel("mean score (0–1)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("v2 paired accuracy, by mode (same structures both modes)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "v2_fig1_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_gap(res, levels, out):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    for m in _models(res):
        xs = [i for i, lv in enumerate(levels) if lv in res[m]]
        ys = [res[m][levels[i]]["gap"] for i in xs]
        ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], lw=2, ms=7, mec="white",
                mew=1.2, label=m)
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylabel("symbolic − content  (paired gap)"); ax.set_xlabel("difficulty")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("v2 de-confounded representation gap  (>0 = symbolic easier)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig(out / "v2_fig2_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig_surface(res, levels, out):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    for m in _models(res):
        xs = [i for i, lv in enumerate(levels) if lv in res[m]]
        ys = [res[m][levels[i]]["flip_rate"] for i in xs]
        ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m], lw=2, ms=7, mec="white",
                mew=1.2, label=m)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylim(0, None); ax.set_ylabel("flip rate across K renderings"); ax.set_xlabel("difficulty")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("v2 surface sensitivity  (structures where correctness flips with phrasing)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig(out / "v2_fig3_surface.png", bbox_inches="tight")
    plt.close(fig)


def _tbl(res, levels, key, fmt="{:.3f}"):
    lines = ["| model | " + " | ".join(f"L{l}" for l in levels) + " |",
             "|---|" + "---|" * len(levels)]
    for m in _models(res):
        cells = [fmt.format(res[m][l][key]) if l in res[m] else "—" for l in levels]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md(res, levels, out):
    md = ["# ArgGYM evals v2 — paired-content benchmark (issue #15)", "",
          "One logical structure per item, rendered both symbolically and in content; "
          "content additionally replicated across K distinct KB records (matched by "
          "Weisfeiler-Leman iso-signature + central-claim gold). Levels {3, 5} "
          "(L1 dropped — its central-claim query is trivially justified/unsatisfiable "
          "with no conflict). Every item queries the central claim c0 and its negation, "
          "scored by the ASPIC+ engine. Content and symbolic are never averaged.", "",
          "## Metric 1 — representation gap (paired, same structure)", "",
          "**Content accuracy** (all renderings):", "", _tbl(res, levels, "content_mean"), "",
          "**Symbolic accuracy** (paired twin):", "", _tbl(res, levels, "symbolic_mean"), "",
          "**Paired gap (symbolic − content)** — representation effect, logical instance held constant:",
          "", _tbl(res, levels, "gap", "{:+.3f}"), "",
          "## Metric 2 — surface variance (K≥2 content renderings)", "",
          "**Flip rate** — fraction of multi-rendering structures where the model is "
          "*not* uniformly correct-or-wrong across phrasings (higher = phrasing/world-"
          "knowledge drives the answer):", "", _tbl(res, levels, "flip_rate"), "",
          "**Mean within-structure score stdev**:", "", _tbl(res, levels, "surface_stdev"), "",
          "## Figures", "",
          "![accuracy](v2_fig1_accuracy.png)", "", "![gap](v2_fig2_gap.png)", "",
          "![surface](v2_fig3_surface.png)", ""]
    (out / "v2-results.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="paired-v2")
    ap.add_argument("--levels", default="3,5")
    ap.add_argument("--out", default="workspace/model-eval-2026-07-21/v2-report")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    data = load(a.sweep)
    res = build(data, levels)
    if not res:
        print(f"no scored runs for sweep '{a.sweep}' yet.")
        return
    fig_accuracy(res, levels, out)
    fig_gap(res, levels, out)
    fig_surface(res, levels, out)
    write_md(res, levels, out)
    print("models:", _models(res))
    for m in _models(res):
        for l in levels:
            if l in res[m]:
                r = res[m][l]
                print(f"  {m} L{l}: content={r['content_mean']:.3f} "
                      f"symbolic={r['symbolic_mean']:.3f} gap={r['gap']:+.3f} "
                      f"flip={r['flip_rate']:.3f} (n_struct={r['n_struct']}, multi={r['n_multi']})")
    print(f"wrote {out}/v2-results.md + figures")


if __name__ == "__main__":
    main()
