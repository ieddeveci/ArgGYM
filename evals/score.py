"""Score a finished run, offline. This program never opens a socket.

Everything it needs is on disk: the completions from `run.py` and the rows from
the frozen taskset. So a scorer fix, a template change or an extraction bug
costs seconds here rather than the hours of inference that produced the
completions.

    uv run python -m evals.score outputs/runs/<dir> --taskset data/taskset.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import arggym
from evals import artifacts, taskset, values
from evals.prompt import region
from evals.types import TIMEOUT

#: The tail of the latency distribution, reported instead of the maximum alone.
#: The maximum is one item -- a provider hiccup moves it and nothing else --
#: while a mean hides the tail entirely, and the tail is the whole question: a
#: timeout is lost inference for the slowest few percent of items, not for the
#: typical one. The maximum is reported beside it rather than instead of it, so
#: a single outlier is visible as an outlier.
#: It is published as `latency_s_p95`, spelled out rather than derived from this
#: constant: the field name is what `report.py` and `results.csv` carry, and a
#: name built at import time would let a changed percentile relabel an existing
#: column instead of adding a new one. Changing the percentile means editing
#: both, which is the point.
LATENCY_PERCENTILE = 0.95

#: How close to `timeout_s` counts as near the wall. A warning has to fire while
#: there is still room to act, and this one has to fire before the first item is
#: lost, because a request that hits the timeout is retried and makes the server
#: generate the whole completion again. Four fifths leaves a quarter of the
#: observed latency as headroom, and is wide enough that ordinary item-to-item
#: variance on a healthy endpoint does not trip it. Nothing here is a threshold
#: anything acts on automatically; it is a number in a report for a person.
NEAR_TIMEOUT_FRACTION = 0.8

#: What an API error with no recorded cause is called. Not a cause: it is the
#: absence of one, and it appears only on generations made before `error_kind`
#: existed. Naming it keeps it visible in the split instead of letting it hide
#: inside whichever real bucket looked closest.
UNCLASSIFIED = "unclassified"

#: How the scorer's `reason` starts when the bloat gate zeroed an answer
#: (`arggym/core/scoring.py`: `bloated:{n_used}_used_vs_{minimum}_minimum`).
BLOATED = "bloated:"


class TasksetMismatch(SystemExit):
    """The run was generated against different questions than these."""


def answer_of(gen: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
    """The text to score, and what had to be done to find it.

    The completion is read first. Only if it holds no answer region is the
    reasoning trace consulted, because a model that reaches an answer mid-thought
    and never submits it has done something different from one that submits, and
    the flag says which. The fallback never changes a score that a submitted
    answer would have got; it only rescues one that was never submitted.
    """
    body, found = region(gen.get("completion") or "", template)
    if found or not gen.get("reasoning"):
        return {"answer": body, "no_answer_region": not found, "answer_in_cot": False}
    merged, found_cot = region(f"{gen['reasoning']}\n{gen.get('completion') or ''}",
                               template)
    return {"answer": merged if found_cot else body,
            "no_answer_region": not found_cot, "answer_in_cot": found_cot}


def score_one(gen: Dict[str, Any], row: Dict[str, Any],
              template: Optional[str]) -> Dict[str, Any]:
    """One scored record. An API error is never a zero."""
    out: Dict[str, Any] = {
        "id": gen["id"], "task": gen["task"], "level": gen["level"],
        "ordering": gen["ordering"], "api_error": gen.get("error"),
        # What kind of failure it was, as `client.py` labelled it at the moment
        # it happened. Absent on generations made before that label existed,
        # which `_error_kinds` reports as `unclassified` rather than guessing
        # the kind back out of the message.
        "api_error_kind": gen.get("error_kind"),
        "truncated": bool(gen.get("truncated")),
        "completion_tokens": (gen.get("usage") or {}).get("completion_tokens"),
        "latency_s": gen.get("latency_s"),
        # The slowest single request, which is what `timeout_s` bounds, and how
        # many of this row's requests expired. The second is the only record of
        # a row that hit the wall and then answered.
        "attempt_latency_s": gen.get("attempt_latency_s"),
        "requests_timed_out": gen.get("requests_timed_out"),
        "attempts": gen.get("attempts"),
    }
    # Every documented field is present on every record, whatever happened.
    # `samples.jsonl` is read outside this process, and a reader that has to
    # guess whether a missing key means false or means "not applicable" is
    # being invited to guess wrong.
    out.update(scorer_refused=False, zero_with_region=False)
    if gen.get("error"):
        # No score, no success, no zero. The item was not measured.
        out.update(score=None, success=None, reason="api_error",
                   no_answer_region=None, answer_in_cot=None)
        return out

    by_value = gen.get("value") is not None
    if by_value:
        # A solver that produced the answer as a value did its own parsing, so
        # nothing is extracted and no fence was ever asked for.
        out.update(no_answer_region=False, answer_in_cot=False)
    else:
        found = answer_of(gen, template)
        out.update(no_answer_region=found["no_answer_region"],
                   answer_in_cot=found["answer_in_cot"])

    try:
        result = (arggym.score_row_value(values.decode(gen["value"], row), row)
                  if by_value
                  else arggym.score_row(found["answer"], row))
    except Exception as e:  # noqa: BLE001
        # `score_row` refuses a row it cannot score -- a mismatched engine
        # version, a missing field. That is a fault in the taskset or the
        # install, not an answer worth zero, so it is flagged rather than
        # averaged. The previous harness caught this with a broad except, called
        # it 0.0, and published a cell of forty items scoring exactly 0.000.
        out.update(score=None, success=None, reason=f"{type(e).__name__}: {e}",
                   scorer_refused=True)
        return out

    out.update(score=result.score, success=result.success, reason=result.reason,
               diagnostics=result.diagnostics, scorer_refused=False)
    # A well-formed answer that scores zero is the interesting case: either the
    # model reasoned wrongly, which is a result, or the scorer is wrong, which
    # is a bug. Counting them is how the second gets noticed.
    out["zero_with_region"] = (result.score == 0.0 and not out["no_answer_region"])
    return out


def _stats(records: Sequence[Dict[str, Any]], floor: Optional[float],
           floor_strategy: Optional[str] = None) -> Dict[str, Any]:
    """One group's numbers, with what was not measured said first."""
    n = len(records)
    ok = [r for r in records if r["score"] is not None]
    untrunc = [r for r in ok if not r["truncated"]]
    mean = sum(r["score"] for r in ok) / len(ok) if ok else None
    out: Dict[str, Any] = {
        "n": n, "n_scored": len(ok),
        "n_api_error": sum(1 for r in records if r["api_error"]),
        # Inside `n_api_error`, not beside it. A timeout is the one API error
        # that moves with the token cap, and the August sweep's error rate rose
        # from 9.8% to 21.8% between levels 3 and 9 -- a shape only visible when
        # the count can be sliced by level, which is what this grouping is for.
        "n_api_timeout": _n_timeout(records),
        # And every request that expired, including the ones whose row went on
        # to answer. Sliced by level for the same reason: this is the count that
        # moves first, before any row is lost.
        "n_requests_timed_out": _n_requests_timed_out(records),
        "n_scorer_refused": sum(1 for r in records if r.get("scorer_refused")),
        # Read before the score. On the August sweep truncation removed 74-89%
        # of items for three of seven models, and a mean over what survived is
        # a measurement of the token cap.
        "truncated_rate": _rate(sum(r["truncated"] for r in records), n),
        "no_answer_region_rate": _rate(
            sum(bool(r.get("no_answer_region")) for r in ok), len(ok)),
        "answer_in_cot_rate": _rate(
            sum(bool(r.get("answer_in_cot")) for r in ok), len(ok)),
        "mean": round(mean, 4) if mean is not None else None,
        # The same mean over the generations that were not cut off. Reported
        # beside the raw one, never instead of it: on the previous sweep a 9B
        # model scored 0.598 raw and 0.765 censored, and only the pair says why.
        "mean_untruncated": (round(sum(r["score"] for r in untrunc) / len(untrunc), 4)
                             if untrunc else None),
        # Its own denominator. `mean` is over `n_scored` and this one is not, so
        # printing the two beside a single count says a mean of four items was
        # a mean of forty.
        "n_untruncated": len(untrunc),
        "success_rate": _rate(sum(bool(r["success"]) for r in ok), len(ok)),
        "zero_with_region": sum(bool(r.get("zero_with_region")) for r in ok),
        # Answers the construction scorer zeroed for using more than twice the
        # minimum number of directives (`arggym/core/scoring.py`, BLOAT_FACTOR).
        # On a row whose minimum is 2 the partial-credit band is two directives
        # wide, so this is how much of a zero the gate decided rather than the
        # argumentation. Zero on tasks with no directive budget.
        "bloat_rate": _rate(sum(str(r.get("reason") or "").startswith(BLOATED)
                                for r in ok), len(ok)),
    }
    if floor is not None:
        out["floor"] = round(floor, 4)
        # What reached it, in the artifact rather than only on the terminal.
        # A floor of 0.000 that no strategy could fit and a floor of 0.000 on a
        # task guessing cannot touch are different findings, and the number
        # alone spells them the same way (`arggym/core/floors.py`).
        out["floor_strategy"] = floor_strategy
        # Chance-corrected, which is the only form in which two tasks' scores
        # are the same quantity (`docs/dataset-contract.md` section 10). The
        # floor bounds the mean score and nothing else, so nothing else here is
        # corrected by it.
        out["corrected"] = (round(arggym.corrected(mean, floor), 4)
                            if mean is not None else None)
    return out


def coverage(records: Sequence[Dict[str, Any]],
             timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """What was not measured. No score in here, deliberately.

    A pooled mean over every record is a mean across tasks, which is the one
    number `aggregate` refuses to produce. Publishing it in the file the report
    reads would make the artifact contradict its own docstring, and it is the
    obvious thing for a reader to quote.

    Latency belongs here for the same reason truncation does: it is a property
    of the run rather than of the model, and it says whether the numbers below
    it were measured at all.
    """
    n = len(records)
    ok = [r for r in records if r["score"] is not None]
    return {
        "n": n, "n_scored": len(ok),
        "n_api_error": sum(1 for r in records if r["api_error"]),
        "n_api_timeout": _n_timeout(records),
        # The whole split, at the one level where the long tail of causes is
        # worth printing. A per-task version would be mostly empty columns.
        "api_errors_by_kind": _error_kinds(records),
        "n_scorer_refused": sum(1 for r in records if r.get("scorer_refused")),
        "truncated_rate": _rate(sum(r["truncated"] for r in records), n),
        "no_answer_region_rate": _rate(
            sum(bool(r.get("no_answer_region")) for r in ok), len(ok)),
        "answer_in_cot_rate": _rate(
            sum(bool(r.get("answer_in_cot")) for r in ok), len(ok)),
        **_latency(records, timeout_s),
    }


def _n_timeout(records: Sequence[Dict[str, Any]]) -> Optional[int]:
    """Rows whose every request expired, or `None` when that is not knowable.

    An error with no recorded cause is not a non-timeout. Counting it as one
    prints `0 timeouts` beside `4 API errors` in the table this report's first
    line tells people to read first, which is the same lie as rendering an
    absent count as zero -- one layer further down, where `_int` cannot see it.
    So a single unclassified error makes the count unknown rather than low. The
    full split stays in `api_errors_by_kind`, where `unclassified` is visible
    beside whatever was classified, so refusing the number here loses nothing.
    """
    if any(r["api_error"] and not r.get("api_error_kind") for r in records):
        return None
    return sum(1 for r in records if r.get("api_error_kind") == TIMEOUT)


def _n_requests_timed_out(records: Sequence[Dict[str, Any]]) -> Optional[int]:
    """Requests that hit the wall, across every row, including rows that
    recovered.

    The number `n_api_timeout` cannot give. With the shipped `retries: 2` a row
    is labelled `timeout` only when all three of its requests expired, so a row
    that hit the wall once and answered on the retry has no error, no kind, and
    a latency taken from the attempt that worked -- it is absent from every
    other number here while having cost two full completions of server time.
    That waste is exactly what `conf/config.yaml` sized the timeout to avoid.

    `None` for generations written before the counter existed, for the same
    reason as above: they made requests and nobody counted the expired ones.
    """
    if any(r.get("requests_timed_out") is None for r in records):
        return None
    return sum(r.get("requests_timed_out") or 0 for r in records)


def _error_kinds(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Every API error by cause, counted.

    A record with an error and no kind is `unclassified` rather than guessed at.
    It means the generation predates the label, and reading the cause back out
    of the message would turn the taxonomy into a function of how a provider
    words its errors -- which is the thing `error_kind` exists to stop.
    """
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        if r["api_error"]:
            counts[r.get("api_error_kind") or UNCLASSIFIED] += 1
    return dict(sorted(counts.items()))


def _latency(records: Sequence[Dict[str, Any]],
             timeout_s: Optional[float]) -> Dict[str, Any]:
    """How much of the timeout budget the run actually used.

    Two quantities, because a timeout leaves two different marks, and neither
    substitutes for the other.

    The percentiles are over the rows that produced an answer, so they describe
    the requests a working endpoint makes. A failed row is left out because its
    latency measures something else: a timeout's is the timeout, so a run whose
    every row expired would otherwise report a percentile pinned to the wall and
    call it a measurement; a 401's is milliseconds, and enough of those drag the
    percentile down while nothing is being exercised at all.

    A row that failed *retryably* and then answered is a different case again,
    and it is in here by design. Since `attempt_latency_s` is the slowest
    attempt, a 503 that took most of the budget before failing raises this run's
    percentile even though the retry came back at once -- on a healthy 12-row
    run one such attempt moved p95 from near zero to the length of the 503. That
    is the right reading: the endpoint did hold a request open that long, and
    the next item to do it may not get a retry cheap enough to hide it. It does
    mean a percentile here answers "how long did the slowest request take" and
    not "how long does an answer take", so a single figure moving is worth
    looking at `api_errors_by_kind` beside.

    What that leaves out is exactly what `n_requests_timed_out` catches. A row
    that hit the wall and recovered *is* in the percentiles -- it answered, and
    `attempt_latency_s` is its slowest request, which is the one that expired --
    so these numbers do move before any row is lost. A row that never recovered
    is not, because it is an error. So the percentiles under-report at the far
    end and the count does not, and the count is also per request rather than
    per row: three expired requests on one row are one error and three wasted
    completions.

    `attempt_latency_s` is the slowest single request for a row, which is what
    `timeout_s` bounds. `latency_s` is the whole row and exceeds the timeout
    after one retry, so it is the fallback for older generations only, where it
    overstates a retried row and therefore overstates the risk -- the safe
    direction for a warning.
    """
    answered = (_attempt_latency(r) for r in records if not r["api_error"])
    values_s = sorted(v for v in answered if v is not None)
    out: Dict[str, Any] = {
        # Its own denominator: this is over the answered generations, which is
        # neither `n` nor `n_scored` -- a row the scorer refused still made a
        # request and still took time, and a solver with no clock in it
        # contributes no latency at all rather than a zero.
        "n_latency_measured": len(values_s),
        "latency_s_p50": _percentile(values_s, 0.5),
        "latency_s_p95": _percentile(values_s, LATENCY_PERCENTILE),
        "latency_s_max": round(values_s[-1], 1) if values_s else None,
        # Repeated from the generations so the percentile above can be read
        # without opening another file. A latency of 2300s is comfortable under
        # 5400 and over the wall under 1800, and the number alone says neither.
        "timeout_s": timeout_s,
        "n_requests_timed_out": _n_requests_timed_out(records),
    }
    near = None if timeout_s is None else timeout_s * NEAR_TIMEOUT_FRACTION
    out["near_timeout_s"] = round(near, 1) if near is not None else None
    out["n_near_timeout"] = (None if near is None
                             else sum(1 for v in values_s if v >= near))
    return out


def _attempt_latency(record: Dict[str, Any]) -> Optional[float]:
    v = record.get("attempt_latency_s")
    return record.get("latency_s") if v is None else v


def _timeout_of(generations: Sequence[Dict[str, Any]],
                run: Dict[str, Any]) -> Optional[float]:
    """The deadline these generations were made under, or `None` if not one
    deadline.

    Read from the generations rather than from the manifest, because the two can
    disagree. `run.py` rewrites `run.json` on every invocation and
    `refuse_a_changed_run` deliberately does not compare `timeout_s` -- a
    changed timeout is not a changed question, so refusing to resume over it
    would cost a sweep its generations for no scientific reason. That was
    harmless while nobody reported the timeout. It is not harmless now: a run
    resumed under 5400 would have its 900s-era generations measured against a
    4320s line and reported as comfortable.

    So the yardstick comes from the same record as the latency it measures, and
    a directory holding generations from two deadlines reports no headroom
    rather than the wrong headroom. The manifest is the fallback for
    generations written before the deadline was recorded per request.
    """
    seen = {g["request"]["timeout_s"] for g in generations
            if isinstance(g.get("request"), dict)
            and g["request"].get("timeout_s") is not None}
    if len(seen) == 1:
        return seen.pop()
    if seen:
        return None
    return (run.get("endpoint") or {}).get("timeout_s")


def _percentile(values_s: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank, so every value reported is a latency something actually had.

    Interpolating between two items would report a duration no request took,
    which is a poor thing to size a timeout against.
    """
    if not values_s:
        return None
    i = min(len(values_s) - 1, max(0, math.ceil(q * len(values_s)) - 1))
    return round(values_s[i], 1)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _floors_of(rows: Sequence[Dict[str, Any]], fit_rows: Sequence[Dict[str, Any]]
               ) -> Tuple[Dict[str, float], Dict[str, str], Optional[str]]:
    """Measured floors, what reached them, and why there are none when there are none.

    Measured over `rows`, the ones this group scored; fitted over `fit_rows`, the
    whole taskset. A fitted strategy searches a map, and a map searched against
    the labels of the ten rows it then corrects is optimistic by hindsight -- up
    to 0.040 on `semantics_query`, on three of its ten reporting groups. The map
    belongs to the dataset, the number to the rows (`arggym/core/floors.py`).

    `arggym.floors` scores every row with every strategy and does not guard, so
    a single ungradeable row would abort the whole scoring pass --
    after `score_one` has already recorded that row politely. Losing every
    number to one bad row is the wrong trade.

    The reason comes back with the result rather than being dropped. Without it
    a group whose floor could not be measured is written exactly like a group
    that has no floor at all, and the report renders both as `-`: one ungradeable
    row would quietly remove a task's `corrected` column with nothing saying so.
    """
    try:
        got = arggym.floors(list(rows), fit_rows=list(fit_rows))
        return ({t: v["floor"] for t, v in got.items()},
                {t: arggym.floor_strategy(v) for t, v in got.items()}, None)
    except Exception as e:  # noqa: BLE001 - recorded on the group and in _meta
        return {}, {}, f"{type(e).__name__}: {e}"


def aggregate(records: List[Dict[str, Any]], rows: Dict[str, Dict[str, Any]]
              ) -> Dict[str, Any]:
    """Per task, per task and level, per task and ordering. Never one mean.

    There is deliberately no overall average. The twelve metrics are of four
    kinds and their chance floors span half the range, so a mean over them moves
    mostly with which tasks are in the basket. The previous harness published one
    as a "ranking aid"; the contract says not to.

    Each grouping measures its own floor over its own rows. A floor is measured,
    not constant: `docs/dataset-contract.md` section 10 quotes them per level
    because that is what they vary with, so correcting a level-15 mean by a
    floor averaged over levels 3 to 15 produces a well-formed number that means
    nothing.
    """
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "by_task": defaultdict(list), "by_task_level": defaultdict(list),
        "by_task_ordering": defaultdict(list)}
    for r in records:
        groups["by_task"][r["task"]].append(r)
        groups["by_task_level"][f"{r['task']}|L{r['level']}"].append(r)
        groups["by_task_ordering"][f"{r['task']}|{r['ordering']}"].append(r)

    out: Dict[str, Any] = {}
    for name, buckets in groups.items():
        out[name] = {}
        for key, got in sorted(buckets.items()):
            mine = [rows[r["id"]] for r in got if r["id"] in rows]
            floors, strategies, why = _floors_of(mine, list(rows.values()))
            task = key.split("|")[0]
            floor = floors.get(task)
            out[name][key] = _stats(got, floor, strategies.get(task))
            if floor is None and why:
                out[name][key]["floor_error"] = why
    return out


def score_run(run_dir: str, taskset_path: Optional[str] = None) -> Dict[str, Any]:
    with open(os.path.join(run_dir, artifacts.RUN)) as f:
        run = json.load(f)
    path = taskset_path or run["taskset"]
    ts_manifest, all_rows = taskset.load(path)

    got, want = ts_manifest.get("taskset_hash"), run.get("taskset_hash")
    if got != want:
        raise TasksetMismatch(
            f"{path} has taskset_hash {got}, and this run was generated against "
            f"{want}. Scoring answers against questions they were not asked is a "
            f"silent way to publish a wrong number; find the taskset this run "
            f"used, or rerun it.")

    rows = {r["id"]: r for r in all_rows}
    generations = artifacts.best_per_id(run_dir)
    unknown = sorted(set(generations) - set(rows))
    if unknown:
        raise TasksetMismatch(
            f"{len(unknown)} generated ids are in no row of {path} "
            f"({', '.join(unknown[:3])}).")

    if "template" not in run:
        # `null` is a legitimate template -- it means the whole completion is
        # the answer -- so an absent key cannot be read as one. Guessing would
        # hand the strict parser a reasoning preamble and report a plausible
        # low score with no warning.
        raise TasksetMismatch(
            f"{artifacts.RUN} in {run_dir} names no template. It is written by "
            f"run.py and is needed to read the answer back out; without it "
            f"there is no way to tell 'no fence was asked for' from 'the key is "
            f"missing', and the two score differently.")
    template = run["template"]
    records = [score_one(generations[i], rows[i], template) for i in sorted(generations)]

    artifacts.write_jsonl(os.path.join(run_dir, artifacts.SAMPLES), records)

    grouped = aggregate(records, rows)
    metrics = {
        "_meta": {
            "run_dir": os.path.abspath(run_dir),
            "endpoint": run.get("endpoint"), "template": template,
            "elicitation": run.get("elicitation"),
            "taskset": path, "taskset_hash": got,
            "taskset_versions": ts_manifest.get("versions", {}),
            "run_status": run.get("status"),
            "arggym": arggym.__version__,
            # A floor moves with the search as well as with the scorer, and
            # `scoring_version` sees only the scorer. Without this, two of these
            # files carrying different floors for the same rows are identical in
            # every field they hold.
            "floors_version": arggym.FLOORS_VERSION,
            "n_selected": run.get("n_selected"), "n_scored": len(records),
            # A run interrupted before it finished has fewer generations than
            # it selected, and this is the only sign of it once `run.json` is
            # out of view. Taken from the count `run.py` made by id where there
            # is one: subtracting two set sizes goes negative as soon as the
            # directory holds an id this run did not select.
            "n_not_generated": (run["n_missing"] if run.get("n_missing") is not None
                                else max(0, (run.get("n_selected") or len(records))
                                         - len(records))),
            # Groups whose chance floor could not be measured, so a `-` in the
            # floor column is never read as "this task has no floor".
            "floors_unmeasured": sorted({
                v["floor_error"] for g in ("by_task", "by_task_level",
                                           "by_task_ordering")
                for v in grouped[g].values() if v.get("floor_error")}),
        },
        # What was not measured, before anything that was. The timeout comes
        # from the generations themselves, falling back to the manifest: this
        # program never builds an `Endpoint` and never reads the config the run
        # was launched with, so a default here would be a guess about a run
        # someone else configured.
        "coverage": coverage(records, _timeout_of(generations.values(), run)),
        **grouped,
    }
    artifacts.write_json(os.path.join(run_dir, artifacts.METRICS), metrics)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--taskset", default=None,
                   help="Defaults to the path the run recorded.")
    a = p.parse_args(argv)
    m = score_run(a.run_dir, a.taskset)

    c = m["coverage"]
    print(f"{c['n_scored']} scored, {c['n_api_error']} API errors, "
          f"{c['n_scorer_refused']} unscorable, {_pct(c['truncated_rate'])} truncated, "
          f"{_pct(c['no_answer_region_rate'])} with no answer region, "
          # Only ever raises a score: it credits an answer the model reached
          # mid-thought and never submitted. A reader deciding whether a number
          # is a measurement of argumentation needs to see how often that
          # happened.
          f"{_pct(c['answer_in_cot_rate'])} answered only in the reasoning\n")
    # Said on the terminal as well as in the file. The point of the number is to
    # be read before a sweep loses items to it, and nobody opens metrics.json to
    # check a thing they have no reason to suspect yet.
    print(f"slowest request {_duration(c['latency_s_max'])}, "
          f"p95 {_duration(c['latency_s_p95'])} "
          f"over {c['n_latency_measured']} answered "
          f"(timeout {_duration(c['timeout_s'])}, "
          f"{_count(c['n_near_timeout'])} within "
          f"{NEAR_TIMEOUT_FRACTION:.0%} of it)")
    # Printed whatever it is, including zero. The percentiles above miss every
    # row that expired on all of its attempts, and count a row rather than the
    # requests it burned; this line misses neither.
    print(f"requests that hit the timeout: {_count(c['n_requests_timed_out'])}; "
          f"rows lost to it: {_count(c['n_api_timeout'])}")
    if c["api_errors_by_kind"]:
        print("API errors by cause: "
              + ", ".join(f"{k} {v}" for k, v in c["api_errors_by_kind"].items()))
    print()
    if m["_meta"]["floors_unmeasured"]:
        print(f"floors could not be measured for some groups "
              f"({'; '.join(m['_meta']['floors_unmeasured'])}); their `corrected` "
              f"column is absent rather than zero.\n")
    print(f"{'task':26s} {'n':>4s} {'scored':>6s} {'mean':>7s} {'untrunc':>8s} "
          f"{'floor':>7s} {'corr':>7s} {'succ':>7s} {'bloat':>7s}")
    for task, v in m["by_task"].items():
        print(f"{task:26s} {v['n']:4d} {v['n_scored']:6d} {_f(v['mean'])} "
              f"{_f(v['mean_untruncated'])} {_f(v.get('floor'))} "
              f"{_f(v.get('corrected'))} {_f(v['success_rate'])} "
              f"{_f(v['bloat_rate'])}")
    print("\nNo overall mean: the per-task metrics are not the same quantity.")
    return 0


def _f(x: Optional[float]) -> str:
    return "      -" if x is None else f"{x:7.3f}"


def _pct(x: Optional[float]) -> str:
    """A rate with no denominator has no percentage, and says so."""
    return "-" if x is None else f"{x:.1%}"


def _duration(x: Optional[float]) -> str:
    """Unknown reads as unknown. A run whose generations never recorded a
    timeout has no distance to the wall, and printing `0s` would claim it had.

    Named apart from `report.py`'s `_secs` because the two differ: this writes
    prose for a terminal and that writes a markdown cell, so an unknown is a
    word here and a dash there. Two helpers with one name and two behaviours is
    how a reader ends up trusting the wrong one.
    """
    return "unknown" if x is None else f"{x:,.0f}s"


def _count(x: Optional[int]) -> str:
    """Same rule for a count: nobody measured it is not the same as zero."""
    return "not recorded" if x is None else str(x)


if __name__ == "__main__":
    raise SystemExit(main())
