"""Draw finished runs. Reads `metrics.json` and `samples.jsonl`; runs nothing.

    uv run --extra report python -m evals.figures outputs/runs/* -o outputs/figures

The same layer as `evals/report.py`, drawn instead of tabulated, and it reads the
same artifacts through the same loader so a column of the table and a point on a
curve are the same number. Five figures, in the order a reader needs them:

  1 contamination        — fraction of items lost to truncation or an API error,
                           by level. FIRST because it decides which of the other
                           numbers mean anything.
  2 macro_by_level       — macro-average vs difficulty, contaminated runs drawn
                           dashed so an eye cannot mistake them for measurements.
  3 recognise_vs_construct — status_query against the mean of the six
                           construction tasks.
  4 task_profile         — per-task x level for the least contaminated run,
                           apples-to-apples.
  5 tokens_vs_truncation — output length against truncation rate, the mechanism
                           behind figure 1.

No figure aggregates a score across tasks without labelling it a macro-average,
and no figure puts a contaminated cell on the same visual footing as a clean one.

Two runs on different tasksets are refused rather than drawn together: they
answered different questions, so a line through both compares models on
different exams. `report.py` prints that as a warning because a table names its
rows; a curve does not, so here it is an error.

Coverage and token length are counted from `samples.jsonl` rather than read from
`metrics.json`. `coverage` there is one block for the whole run, and these
figures need it per level; completion tokens are not aggregated into
`metrics.json` at all. Scores are taken from `metrics.json` so that a point and
the corresponding cell of `report.py`'s table cannot disagree.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - must follow the Agg backend choice

from evals import artifacts, report  # noqa: E402 - same

#: Six tasks answered with directives -- the ones that ask the model to *build*
#: something rather than read off a status.
CONSTRUCTION = ("attack", "defence", "attack_defense", "counter_argument",
                "counter_argument_strict", "preference_construction")

#: Above this fraction lost to truncation + API error, a cell's score is not a
#: reasoning measurement. Drawn dashed throughout so the distinction is visible
#: without reading a caption.
CONTAM_LIMIT = 0.20

#: Nine, against `tab10`'s ten colours. Ten would share a period with the
#: colour cycle, so an eleventh run would come back identical to the first --
#: replacing a roster that silently dropped unknown models with one that
#: silently merges runs. Coprime, the pair repeats after 90.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<")


class NoLevels(SystemExit):
    """A run carries no per-level grouping, so there is no x-axis to draw."""


class Incomparable(SystemExit):
    """Runs answering different questions were asked for in one figure."""


class TooMany(SystemExit):
    """More runs than the palette can draw distinguishably."""


class Run:
    """One scored run directory: its label, its metrics, its records.

    The directory comes from the command line and not from `_meta.run_dir`. That
    field is an absolute path written when the run was scored, so it names the
    wrong machine as soon as a run directory is copied off the box that produced
    it -- which is the normal way these files travel.
    """

    def __init__(self, path: str, metrics: Dict[str, Any]) -> None:
        self.dir = path if os.path.isdir(path) else os.path.dirname(path)
        self.metrics = metrics
        self.label = metrics["_label"]
        samples = os.path.join(self.dir, artifacts.SAMPLES)
        if not os.path.exists(samples):
            raise report.NotScored(
                f"{self.dir} has {artifacts.METRICS} but no {artifacts.SAMPLES}. "
                f"Both are written by one call to `evals.score`, so a directory "
                f"with one and not the other was assembled by hand; figures 1 "
                f"and 5 are counted per level and per item, which "
                f"{artifacts.METRICS} does not carry.")
        self.records = list(artifacts.read_jsonl(samples))

    @property
    def levels(self) -> List[int]:
        return sorted({r["level"] for r in self.records})

    def mean_at(self, task: str, level: int) -> Optional[float]:
        """The published mean for one (task, level), or None if there is no cell.

        `mean` is subscripted rather than `.get`-ed. A group that exists and has
        lost the field is a schema change in `evals/score.py`, and the whole
        point of this file's repair is that such a change stops the figures
        loudly instead of drawing zeros.
        """
        group = self.metrics["by_task_level"].get(f"{task}|L{level}")
        return None if group is None else group["mean"]

    @property
    def tasks(self) -> List[str]:
        return sorted(self.metrics["by_task"])

    @property
    def cap(self) -> Optional[int]:
        """The token cap, when one bound some generation in this run.

        A cap no generation reached is not drawn. A line at 16,384 through a
        figure whose longest point is 8K says a constraint is in play that is
        not, and figure 5 exists to show the one that is.
        """
        if not any(r["truncated"] for r in self.records):
            return None
        sampling = (self.metrics["_meta"].get("endpoint") or {}).get("sampling") or {}
        # Subscripted. A run that truncated hit a cap by definition, so an
        # endpoint that records sampling and does not name `max_tokens` has
        # renamed it, and `.get` would answer that by quietly dropping the one
        # line this figure exists to draw.
        return sampling["max_tokens"] if sampling else None


def load(paths: Sequence[str]) -> List[Run]:
    """Every named run, refusing a mixed taskset and an unscored directory."""
    metrics = report.load_metrics(list(paths))
    runs = [Run(m["_path"], m) for m in metrics]
    # Subscripted, and this is the field that most needs it: drawing an error
    # rather than printing `report.py`'s warning is the whole design, and a
    # `.get` here disarms it silently the day `evals/score.py` renames the key.
    # Every run would then report `None`, the set would hold one element, and
    # two sweeps on different tasksets would draw one curve.
    hashes = {r.metrics["_meta"]["taskset_hash"] for r in runs}
    if len(hashes) > 1:
        raise Incomparable(
            f"these runs used {len(hashes)} different tasksets "
            f"({', '.join(sorted(str(h) for h in hashes))}). They answered "
            f"different questions, so one curve through them compares models on "
            f"different exams. Draw them separately.")
    if not any(r.levels for r in runs):
        raise NoLevels(
            "no run carries a level on any record, so every figure here would "
            "be a line with no x-axis.")
    return runs


def levels_of(runs: Sequence[Run]) -> List[int]:
    return sorted({lv for r in runs for lv in r.levels})


def macro(run: Run, level: int) -> Optional[float]:
    """Mean of the per-task means at one level, and nothing else.

    `metrics.json` publishes no overall mean on purpose (`evals/score.py`), so
    this is computed here and labelled a macro-average everywhere it is drawn.
    """
    vals = [v for t in run.tasks
            if (v := run.mean_at(t, level)) is not None]
    return sum(vals) / len(vals) if vals else None


def lost(run: Run, level: int) -> Optional[float]:
    """Fraction of a level's items that never reached the scorer."""
    return _rate(run, level, lambda r: bool(r["truncated"]) or bool(r["api_error"]))


def truncated(run: Run, level: int) -> Optional[float]:
    return _rate(run, level, lambda r: bool(r["truncated"]))


def errored(run: Run, level: int) -> Optional[float]:
    return _rate(run, level, lambda r: bool(r["api_error"]))


def _rate(run: Run, level: int, hit: Callable[[Dict[str, Any]], bool]) -> Optional[float]:
    at = [r for r in run.records if r["level"] == level]
    return sum(1 for r in at if hit(r)) / len(at) if at else None


def contamination(run: Run) -> float:
    """The run's own loss rate, over every record it holds."""
    if not run.records:
        return 0.0
    return sum(1 for r in run.records
               if r["truncated"] or r["api_error"]) / len(run.records)


def series(run: Run, levels: Sequence[int],
           fn: Callable[[Run, int], Optional[float]]):
    xs, ys = [], []
    for lv in levels:
        v = fn(run, lv)
        if v is not None:
            xs.append(lv)
            ys.append(v)
    return xs, ys


def _style(run: Run, color: str) -> Dict[str, Any]:
    """Dashed + hollow when the run is mostly contamination.

    Judged over the run rather than per point: a line whose dash pattern changes
    mid-curve is harder to read than one that does not, and figure 1 carries the
    per-level truth for anyone who needs it.
    """
    dirty = contamination(run) > CONTAM_LIMIT
    return {"linestyle": "--" if dirty else "-",
            "alpha": 0.55 if dirty else 1.0,
            "markerfacecolor": "none" if dirty else color}


def palette(runs: Sequence[Run]) -> Dict[str, Dict[str, str]]:
    """A colour and a marker per run label.

    Assigned from the label order rather than from a hardcoded roster: the
    roster the August sweep drew is seven model names that will never appear in
    a run directory again, and a figure tool that silently drops every model it
    was not told about in advance is the failure this file is being repaired
    from.
    """
    cmap = plt.get_cmap("tab10")
    if len(runs) > 10 * len(MARKERS):
        raise TooMany(
            f"{len(runs)} runs, and this palette can tell "
            f"{10 * len(MARKERS)} apart. Past that two lines share a colour and "
            f"a marker, and a reader has no way to know which. Draw them in "
            f"groups.")
    return {r.label: {"color": matplotlib.colors.to_hex(cmap(i % 10)),
                      "marker": MARKERS[i % len(MARKERS)]}
            for i, r in enumerate(runs)}


def fig1_contamination(runs, levels, style, out) -> str:
    """Total loss, then the two mechanisms behind it, side by side.

    Split because they are different failures with different fixes: hitting the
    token cap is the model refusing to stop, while an API error is the request
    failing or exceeding the wall clock. A model contaminated almost entirely by
    the second and barely at all by the first is a different problem from the
    reverse, and a single combined line hides that.
    """
    panels = [("truncated or API error", lost, "total loss"),
              ("truncated", truncated, "hit the token cap"),
              ("API error", errored, "failed / timed out")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, (lab, fn, sub) in zip(axes, panels):
        # Runs that never leave the floor are drawn as markers only, jittered
        # horizontally. Nudging them *vertically* would be a lie -- it invents a
        # rate the run does not have -- whereas an x-jitter moves a point away
        # from a level it does not claim to sit between, and the marker still
        # reads off the y-axis correctly.
        flat = [r.label for r in runs
                if series(r, levels, fn)[0] and max(series(r, levels, fn)[1]) < 0.01]
        for r in runs:
            xs, ys = series(r, levels, fn)
            if not xs:
                continue
            st = style[r.label]
            if r.label in flat:
                j = 0.28 * (flat.index(r.label) - (len(flat) - 1) / 2)
                ax.plot([x + j for x in xs], [y * 100 for y in ys],
                        marker=st["marker"], color=st["color"], label=r.label,
                        linestyle="none", markersize=5)
            else:
                ax.plot(xs, [y * 100 for y in ys], marker=st["marker"],
                        color=st["color"], label=r.label, linewidth=2, markersize=5)
        ax.axhline(CONTAM_LIMIT * 100, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("level")
        ax.set_title(f"{lab}\n({sub})", fontsize=10)
        ax.set_xticks(levels)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of items")
    axes[0].set_ylim(-3, 100)
    axes[2].text(levels[-1] + 0.4, CONTAM_LIMIT * 100, "unusable\nabove here",
                 fontsize=8, va="center")
    axes[0].legend(fontsize=7.5, loc="center right")
    fig.suptitle("1. Contamination by level — read this before any score "
                 "(runs flat at 0% shown as markers, jittered sideways)",
                 fontsize=11)
    return _save(fig, out, "fig1_contamination.png")


def fig2_macro_by_level(runs, levels, style, out) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    n_tasks = _counted(len(r.tasks) for r in runs)
    for r in runs:
        xs, ys = series(r, levels, macro)
        if not xs:
            continue
        st = style[r.label]
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=r.label,
                linewidth=2, **_style(r, st["color"]))
    ax.set_xlabel("level")
    ax.set_ylabel(f"macro-average (mean of {n_tasks} per-task means)")
    ax.set_title("2. Score vs difficulty\n"
                 f"dashed = >{CONTAM_LIMIT:.0%} of items never reached the scorer",
                 fontsize=11)
    ax.set_xticks(levels)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out, "fig2_macro_by_level.png")


def fig3_recognise_vs_construct(runs, levels, style, out) -> str:
    """The boundary the August sweep drew sharpest: reading a theory vs building in it."""
    def rec(run, lv):
        return run.mean_at("status_query", lv)

    def con(run, lv):
        vs = [v for t in CONSTRUCTION if (v := run.mean_at(t, lv)) is not None]
        return sum(vs) / len(vs) if vs else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for r in runs:
        st = style[r.label]
        line = _style(r, st["color"])
        xs, ys = series(r, levels, rec)
        if xs:
            a1.plot(xs, ys, marker=st["marker"], color=st["color"], label=r.label,
                    linewidth=2, **line)
        xs, ys = series(r, levels, con)
        if xs:
            a2.plot(xs, ys, marker=st["marker"], color=st["color"], linewidth=2,
                    **line)
    a1.set_title("recognise: status_query")
    a2.set_title(f"construct: mean of {_construction_count(runs)} construction tasks")
    for a in (a1, a2):
        a.set_xlabel("level")
        a.set_xticks(levels)
        a.set_ylim(0, 1.0)
        a.grid(alpha=0.3)
    a1.set_ylabel("mean score")
    a1.legend(fontsize=7)
    fig.suptitle("3. Recognising a theory is not constructing in one "
                 "(dashed = contaminated)", fontsize=11)
    return _save(fig, out, "fig3_recognise_vs_construct.png")


def fig4_task_profile(run, levels, out) -> str:
    """One run, every task, every level.

    The least contaminated run, because this is the figure that invites a
    per-task reading and a profile of a run that lost most of its items is a
    profile of the token cap. Which run it is, and how clean it is, goes in the
    title rather than a caption someone has to find.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(run.tasks):
        xs, ys = series(run, levels, lambda r, lv, t=t: r.mean_at(t, lv))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, color=cmap(i % 20),
                    label=t, linewidth=1.6)
    ax.set_xlabel("level")
    ax.set_ylabel("mean score")
    ax.set_title(f"4. Task profile — {run.label} "
                 f"(contamination {contamination(run):.1%})", fontsize=10)
    ax.set_xticks(levels)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    return _save(fig, out, "fig4_task_profile.png")


def fig5_tokens_vs_truncation(runs, levels, style, out) -> str:
    """The mechanism: runs that generate near the cap truncate.

    One point per (run, level). Mean completion tokens is over the records that
    report a token count, which excludes the API errors -- a request that never
    returned has no length, and counting it as zero would pull a run's mean
    towards the origin in proportion to how badly it failed.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    xmax = 0.0
    for r in runs:
        pts = []
        for lv in levels:
            at = [x for x in r.records if x["level"] == lv]
            toks = [x["completion_tokens"] for x in at
                    if x["completion_tokens"] is not None]
            if not at or not toks:
                continue
            pts.append((sum(toks) / len(toks),
                        100 * sum(1 for x in at if x["truncated"]) / len(at)))
        if not pts:
            continue
        xmax = max(xmax, *[p[0] for p in pts])
        st = style[r.label]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=st["color"],
                   marker=st["marker"], s=70, label=r.label, edgecolors="black",
                   linewidths=0.4)
    for cap in sorted({r.cap for r in runs if r.cap is not None}):
        xmax = max(xmax, cap)
        ax.axvline(cap, color="#555555", linestyle=":", linewidth=1)
        ax.text(cap - 0.012 * max(xmax, 1), 45, f"cap = {cap:,}", fontsize=8,
                rotation=90, va="center", ha="right", color="#555555")
    ax.set_xlabel("mean completion tokens")
    ax.set_ylabel("% truncated")
    ax.set_xlim(0, max(xmax, 1) * 1.10)
    ax.set_ylim(-4, 100)
    ax.set_title("5. Truncation tracks output length\n"
                 "(only a cap some generation actually hit is drawn)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out, "fig5_tokens_vs_truncation.png")


def _counted(per_run) -> str:
    """How many tasks a title may claim, when the runs disagree about it.

    `macro` and the construction mean are computed per run, so a count taken as
    the union across runs describes no line on the chart. A 2-item run drawn at
    1.0 under a title saying "mean of 6 construction tasks" is a mean of one,
    sitting above every honest run -- the same quiet wrong number this file is
    being repaired from, reintroduced in the label instead of the data.
    """
    counts = sorted(set(per_run))
    return str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"


def _construction_count(runs: Sequence["Run"]) -> str:
    return _counted(sum(t in r.tasks for t in CONSTRUCTION) for r in runs)


def _save(fig, out: str, name: str) -> str:
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def profile_of(runs: Sequence[Run]) -> Run:
    """Which run figure 4 profiles: least contaminated, then largest.

    Contamination alone ties, and often. Three of the four runs this was first
    checked against sit at exactly 0.0 -- one of them holds two records, one is
    a stub -- so the tie broke on whatever order the labels happened to sort
    into, and a profile of two items is not a profile.
    """
    return min(runs, key=lambda r: (contamination(r), -len(r.records)))


def draw(runs: Sequence[Run], out: str) -> List[str]:
    os.makedirs(out, exist_ok=True)
    levels = levels_of(runs)
    style = palette(runs)
    profile = profile_of(runs)
    return [fig1_contamination(runs, levels, style, out),
            fig2_macro_by_level(runs, levels, style, out),
            fig3_recognise_vs_construct(runs, levels, style, out),
            fig4_task_profile(profile, levels, out),
            fig5_tokens_vs_truncation(runs, levels, style, out)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Run directories, or metrics.json paths.")
    p.add_argument("-o", "--out", default="outputs/figures")
    a = p.parse_args(argv)

    runs = load(a.runs)
    print(f"{len(runs)} runs, levels {levels_of(runs)}: "
          f"{', '.join(r.label for r in runs)}")
    for path in draw(runs, a.out):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
