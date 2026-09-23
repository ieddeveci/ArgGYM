"""Freeze a taskset from a spec.

The generator is pure: `ds[k]` is the item at seed `start + k`, and a seed that
builds nothing raises. Densifying is this module's job, because this is where it
can be recorded.

So a cell is filled by scanning seeds until it has what the spec asked for, and
the manifest records both halves: which seeds produced the items, and which were
skipped. That pair is a better determinism check than comparing hashes. Two
exports with the same skips and different hashes mean a renderer or a scorer
moved; different skips mean a generator moved. A hash alone cannot tell you
which.

Under the skip list sits the retry loop, and it is where the interesting number
lives. On the standard grid five seeds of 7205 are skipped, while `perturbation`
at level 6 reaches its ten items by discarding 34 candidates of 44. So
the cell also records how many candidates `build` was asked for and why it
refused them (#72), counted over every seed rather than only the failed ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from arggym.core.build import reasons_text, top_reason
from arggym.core.dataset import TaskDataset
from arggym.core.prompting import MINIMALITY
from arggym.core.rows import score as score_row
from arggym.core.spec import TasksetSpec, check_versions

#: Bumped when the row a harness reads changes shape or wording: the rendered
#: question for any reason, and the metadata schema too. Either half moves it on
#: its own. Version 5 removed `answer_template`, which moved `taskset_hash`
#: without moving a single question; versions 6 and 7 each state rules the scorer
#: was already enforcing, and so move questions and no metadata at all -- the 280
#: engine-scored and `formalization` questions at 6, the remaining 200 at 7.
#: "The rendered question for any reason" covers a generator that writes a different
#: theory as much as a renderer that words the same one differently.
#:
#: Version 8 was set twice, for two disjoint sets of questions, which is the drift
#: this constant exists to prevent (#161). #149 set it for #121, which stopped
#: `formalization` writing a second route to a single queried claim and moved three
#: of its 40 rows. #166 then moved 80 more -- all 40 `formalization` and all 40
#: `defeat_diagnosis`, the first gaining an AND-conjunction clause in
#: `formalization_notation` and the second a paragraph naming what each diagnosis
#: field holds -- and left the number at 8. A run recorded as `prompt_version: 8`
#: does not say which of the two it saw.
#:
#: Version 9 moves `semantics_query`: the share cap at `semantics_query.py:650`
#: compared the largest status in the row rather than the candidate's own, so rows
#: stalled at five queries (#171), and the coverage pass now spends its slot on a
#: claim that splits where one is available (#138). Both change which claims a row
#: asks, on all 120 of its rows. No other task moves.
#:
#: Version 10 moves `counter_argument` and its strict arm: every item now builds at
#: least two target-reaching chains, and the strict arm no longer draws zero strict
#: rules at levels 3 to 5 (#167). That rewrites the theory of both arms at levels 1 and
#: 2 and the strict arm's theory on the 14 cells at levels 3 to 5 where the draw fell,
#: 46 rows in all.
#:
#: Version 11 moves `semantics_query`: the unattacked ring's ordering is `level % 4`
#: rather than `(level // 3) % 4`, so each ordering carries it on levels four apart
#: instead of three levels running (#168). The ring moves on 14 cells at levels 7 to
#: 10 and 13 to 15 and rewrites the theory and gold of their 28 rows. No other task
#: moves.
#:
#: Version 12 moves `semantics_query`: the status share is capped within each semantics
#: an item asks four times or more, rather than over the whole item (#104). Of its 600
#: rows on the standard grid, 522 ask different queries and 120 carry a different theory,
#: because the retry loop accepts a different candidate for that seed. No other task
#: moves.
#:
#: Version 13 moves the five tasks whose prompt carries `permitted_block`: `attack`,
#: `defence`, `attack_defense`, `counter_argument` and `counter_argument_strict`. The
#: sentence on unreadable directives now ends ", so write only directives", as it
#: already did in the blocks of `preference_construction` and `formalization` (#187).
#: Theories, ids and gold are unchanged; only the question's wording moves.
PROMPT_VERSION = 13
#: Bumped when a scoring policy constant or weight changes. Those move a score
#: without moving a prompt, so the hash over prompts cannot see them.
#:
#: A legality rule that the prompt states and the scorer enforces bumps both, the
#: way the bloat factor does (`docs/dataset-contract.md`): the prompt hash records
#: that the question changed, and this records that an unchanged question is now
#: scored differently. #89's name-collision rule is such a rule.
#:
#: Version 5 is #166, and it moves two scorers in opposite directions.
#:
#: `formalization.score_value` compares structural keys from `_alpha_shape_keys`
#: rather than identifier text, so an answer's rule names no longer have to match
#: the reference's -- including where a name is referenced, in an undercut or a rule
#: preference. That admits answers version 4 refused.
#:
#: `defeat_diagnosis` tightens. A survival explanation earns credit only where its
#: failure point was matched including kind (`defeat_diagnosis.py:546-553`);
#: `exact_match` now requires set equality of the `(defeated_at, defeater,
#: survives_because)` triples rather than a survival score above a threshold
#: (`:591-595`); and supplying `survives_because` where the gold carries none is a
#: hard zero, `unexpected_survival_reason` (`:534-542`). That refuses answers
#: version 4 scored. The prompt gained a paragraph naming what each field holds in
#: the same change, which is `PROMPT_VERSION` 8's second meaning -- so a row scored
#: under 5 was also asked differently, and the two constants have to be read
#: together for anything measured before 2026-09-22.
#:
#: Version 6 moves `formalization.score_value` twice, the construction scorer once,
#: every parser once, and `defeat_diagnosis` once more.
#:
#: `success` requires the answer's theory to give the same status as the reference to
#: every atom the reference names, in both polarities, `UNSATISFIABLE` included
#: (#190). It used to check only the queried literals, whose statuses the question
#: prints, so an answer built from those printed statuses succeeded on every row
#: without formalising the text, and an undercut written as a rebuttal usually
#: succeeded too. Scores do not move for this change.
#:
#: `type_score` is the F1 of the axiom and strict decisions, where it used to be their
#: recall (#191). An axiom or strict directive the gold does not have now costs
#: precision, so promoting every premise to an axiom no longer keeps a type score of
#: 1.0. Rows whose gold has no axiom and no strict rule still carry no type term.
#:
#: Partial credit on the six construction tasks pays for movement from each goal's
#: starting status, not for the final theory (#188). A goal the answer left at the
#: wanted status it started at leaves the average, and one it broke scores 0.0;
#: `UNDECIDED` earns its 0.4 only where the goal did not start there; a subgoal counts
#: only if it started `JUSTIFIED` or the answer moved it. An answer that changes
#: nothing now scores 0.0 where it earned up to 0.23. This part raises no score: a
#: success scores as before, and a failed answer scores the same or lower.
#:
#: Every parser drops markdown code markup before it reads (#187, `scoring.unmark`):
#: a line that is only a code fence, with or without a language tag, and every
#: backtick. A correct answer wrapped in a fence or written with each line in
#: backticks scored 0.0 on all 12 tasks under 5 and scores what the bare answer
#: scores under 6. A stray word, backticked or not, still zeroes the answer. This
#: part only admits answers.
#:
#: `defeat_diagnosis` reads the `defeated_at` key in any case, as it already read the
#: other keys, and compares the `defeated_at`, `defeater` and `survives_because`
#: values without regard to case (#192). It used to drop every record whose
#: `defeated_at` was capitalised and score the answer on its status line alone, 0.15
#: with reason `ok`. A lowercase answer scores as before.
SCORING_VERSION = 6


@dataclass
class CellReport:
    task: str
    level: int
    ordering: str
    seeds_used: List[int] = field(default_factory=list)
    seeds_skipped: List[Dict[str, Any]] = field(default_factory=list)
    scan_end: int = 0
    #: Candidates `build` was asked for across every seed scanned, the accepted
    #: ones included, and why each one that did not become a row was refused.
    #: Most reasons are `build`'s own; `minimality_unproven` is the freeze
    #: refusing a candidate `build` returned (#124). Both sit in one histogram, so
    #: a reader asking what the cell threw away reads one number.
    build_calls: int = 0
    build_rejections: Dict[str, int] = field(default_factory=dict)

    @property
    def acceptance(self) -> float:
        tried = len(self.seeds_used) + len(self.seeds_skipped)
        return len(self.seeds_used) / tried if tried else 0.0

    @property
    def build_acceptance(self) -> float:
        """Items per candidate built, which is the rate a thin cell shows up in.

        Seed acceptance hides the retry loop: every cell of the grid fills, and
        `perturbation` at level 6 under `last_link_elitist` still reaches it by
        discarding 34 candidates out of 44 at `take: 10`.
        """
        return len(self.seeds_used) / self.build_calls if self.build_calls else 0.0

    def key(self) -> str:
        return f"{self.task}|L{self.level}|{self.ordering}"

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for s in self.seeds_skipped:
            counts[s["reason"]] = counts.get(s["reason"], 0) + 1
        return {"n": len(self.seeds_used), "seeds_used": self.seeds_used,
                "seeds_skipped": self.seeds_skipped, "reason_counts": counts,
                "scan_end": self.scan_end, "acceptance_rate": round(self.acceptance, 4),
                "build_calls": self.build_calls,
                "build_rejections": dict(sorted(self.build_rejections.items())),
                "build_acceptance_rate": round(self.build_acceptance, 4)}


class CellUnfilled(SystemExit):
    """A cell could not produce what the spec asked for."""


class PromptClaimUnsupported(SystemExit):
    """A question states a rule the row gives the scorer no way to apply."""


class ReferenceNotVerified(SystemExit):
    """A row's own reference answer does not score 1.0 against it."""


def reference_score(entry: Dict[str, Any]) -> float:
    """What the row's reference answer scores against the row.

    Scored through the row rather than through the item, so this is also the
    check that the row carries everything its scorer reads: a row that lost a
    field either refuses or scores its own gold below 1.0, and either way the
    freeze stops. The pre-contract export computed the same number and counted
    how many reached 1.0; a count nobody reads is a check that has already been
    lost, so this refuses instead.
    """
    return score_row(entry["reference_answer"], entry).score


def minimum_unproven(entry: Dict[str, Any]) -> bool:
    """Whether the row's stated minimum is a bound the search gave up on.

    The six construction generators write `minimality_proven` into the item's
    statistics, which the row carries under `metadata.gold.metadata`. Only an
    explicit `False` counts: a task with no minimum search writes nothing there,
    and nothing is not a failed proof.
    """
    stats = (entry.get("metadata", {}).get("gold", {}) or {}).get("metadata") or {}
    return stats.get("minimality_proven") is False


def fill_cell(task: str, level: int, ordering: str, spec: TasksetSpec
              ) -> Tuple[List[Dict[str, Any]], CellReport]:
    """Scan seeds until the cell has `take` items, or fail naming the cell."""
    report = CellReport(task, level, ordering)
    rows: List[Dict[str, Any]] = []
    ds = TaskDataset(task, level, ordering, size=spec.seeds.scan_limit,
                     seed=spec.seeds.start, profile=spec.profile)
    for k in range(spec.seeds.scan_limit):
        if len(rows) == spec.seeds.take:
            break
        seed = spec.seeds.start + k
        report.scan_end = seed
        # `build_at` rather than `ds[k]`: the reasons behind a seed that
        # succeeded on its seventh candidate are the ones with a signal in them,
        # and an exception can only carry the reasons from a seed that failed
        # outright.
        entry, built = ds.build_at(k)
        report.build_calls += built.calls
        for reason, n in built.reasons.items():
            report.build_rejections[reason] = report.build_rejections.get(reason, 0) + n
        if entry is None:
            # `reason` names what this seed hit most often, which is what a skip
            # list is read for; the full count is in `build_rejections`.
            report.seeds_skipped.append({"seed": seed, "tries": built.calls,
                                         "reason": top_reason(built.reasons),
                                         "reasons": dict(sorted(built.reasons.items()))})
            continue
        if minimum_unproven(entry):
            # The minimum search ran out of budget and fell back to every
            # candidate it had, so `min_directives` is an upper bound rather than
            # a minimum: on the one such row of the standard grid it said 55
            # where 10 suffice, and the bloat gate admitted 110 directives where
            # it should admit 20 (#124). A reference that scores below 1.0 stops
            # the freeze; a stated minimum 5.5x the real one is the same class of
            # claim, and the flag that says so was read by nothing. The seed is
            # skipped and named, the way a seed that built nothing is.
            reasons = {**built.reasons, "minimality_unproven": 1}
            report.build_rejections["minimality_unproven"] = (
                report.build_rejections.get("minimality_unproven", 0) + 1)
            report.seeds_skipped.append({"seed": seed, "tries": built.calls,
                                         "reason": "minimality_unproven",
                                         "reasons": dict(sorted(reasons.items()))})
            continue
        entry["metadata"]["reference_score"] = reference_score(entry)
        entry["metadata"]["source_index"] = len(rows)
        rows.append(entry)
        report.seeds_used.append(seed)

    if len(rows) < spec.seeds.take:
        raise CellUnfilled(
            f"{report.key()} yielded {len(rows)} of {spec.seeds.take} items within "
            f"{spec.seeds.scan_limit} seeds (acceptance {report.acceptance:.2f}). "
            f"It discarded {report.build_calls - len(rows)} candidates: "
            f"{reasons_text(report.build_rejections)}. Raise seeds.scan_limit, lower "
            f"seeds.take, or fix what the reasons name -- a cell this thin is a "
            f"finding, not a setting.")
    if report.acceptance < spec.min_acceptance:
        raise CellUnfilled(
            f"{report.key()} filled, but took {report.scan_end - spec.seeds.start + 1} "
            f"seeds for {spec.seeds.take} items (acceptance {report.acceptance:.2f}, "
            f"spec asks for {spec.min_acceptance}). A degraded cell should be a "
            f"decision: lower min_acceptance deliberately or fix what the skipped "
            f"seeds name.")
    if report.build_acceptance < spec.min_build_acceptance:
        # The guard above cannot see a thin cell on the standard grid, where
        # nearly every seed builds and the retry loop hides how many candidates
        # it threw away (#114). This one reads the number the retry loop reports.
        raise CellUnfilled(
            f"{report.key()} filled, but kept {len(rows)} of {report.build_calls} "
            f"candidates (build acceptance {report.build_acceptance:.2f}, spec asks for "
            f"{spec.min_build_acceptance}). It discarded: "
            f"{reasons_text(report.build_rejections)}. A cell this thin is a finding, "
            f"not a setting: lower min_build_acceptance deliberately or fix what the "
            f"reasons name.")
    return rows, report


def taskset_hash(rows: List[Dict[str, Any]]) -> str:
    """A digest over everything a model sees and everything it is graded on.

    Metadata is included, so an edit that changes gold produces a new hash
    rather than silently overwriting a frozen taskset under the same name.
    """
    h = hashlib.blake2b(digest_size=16)
    for r in rows:
        h.update(json.dumps({"id": r["id"], "question": r["question"],
                             "reference_answer": r["reference_answer"],
                             "metadata": r["metadata"]},
                            sort_keys=True, default=str).encode())
    return h.hexdigest()


def _versions() -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    import arggym
    from arggym.core.serialize import THEORY_SCHEMA

    try:
        pyarg = _pkg_version("python-argumentation")
    except PackageNotFoundError:  # pragma: no cover - depends on the install
        pyarg = None
    return {"arggym": arggym.__version__, "pyarg": pyarg,
            "prompt_version": PROMPT_VERSION, "scoring_version": SCORING_VERSION,
            "theory_schema": THEORY_SCHEMA,
            "python": sys.version.split()[0],
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>")}


def freeze(spec: TasksetSpec, path: str, verbose: bool = True) -> Dict[str, Any]:
    """Write one JSONL holding a manifest line and every row."""
    check_versions(spec)
    rows: List[Dict[str, Any]] = []
    cells: Dict[str, Any] = {}
    for task, level, ordering in spec.cells:
        got, report = fill_cell(task, level, ordering, spec)
        # source_index is the row's position in the whole file, not in its cell.
        for r in got:
            r["metadata"]["source_index"] = len(rows)
            rows.append(r)
        # The bloat rule is gated on `min_directives` in the scorer and asserted flatly
        # by every prompt that carries it. All six construction generators do set one,
        # but that was a property of six generators rather than a checked invariant, so
        # a prompt could come to promise a rule the scorer would skip.
        unsupported = [r["id"] for r in got
                       if MINIMALITY in r["question"]
                       and not r["metadata"].get("gold", {}).get("min_directives")]
        if unsupported:
            raise PromptClaimUnsupported(
                f"{report.key()}: {len(unsupported)} of {len(got)} rows state the "
                f"minimality rule and carry no min_directives, so the scorer would not "
                f"apply it ({', '.join(unsupported[:3])}). The question would be making "
                f"a promise the row cannot keep.")
        unverified = [r["id"] for r in got
                      if r["metadata"]["reference_score"] < 0.999]
        if unverified:
            raise ReferenceNotVerified(
                f"{report.key()}: the reference answer does not score 1.0 on "
                f"{len(unverified)} of {len(got)} rows ({', '.join(unverified[:3])}). "
                f"Either the row is missing something its scorer reads or the gold "
                f"is wrong; a taskset whose own answers fail is not publishable.")
        cells[report.key()] = report.to_dict()
        if verbose:
            note = "" if report.acceptance == 1.0 else f"  ({report.acceptance:.0%} of seeds accepted)"
            # Printed whenever the cell built more candidates than it kept, which
            # seed acceptance never shows: every grid cell fills, and some fill by
            # discarding six candidates in seven.
            if report.build_calls > len(got):
                note += (f"  [{report.build_calls} candidates, "
                         f"{report.build_acceptance:.0%} kept: "
                         f"{reasons_text(report.build_rejections)}]")
            print(f"  {report.key():54s} {len(got):3d} items{note}")

    manifest = {
        "spec": spec.to_dict(),
        "versions": _versions(),
        "taskset_hash": taskset_hash(rows),
        "n_items": len(rows),
        # Every reference scores 1.0 or the freeze has already raised, so this
        # equals n_items and is written for the reader who wants to see it said.
        "n_verified": sum(1 for r in rows
                          if r["metadata"]["reference_score"] >= 0.999),
        "cells": cells,
        "n_skipped": sum(len(c["seeds_skipped"]) for c in cells.values()),
        # Candidates `build` discarded, over every seed including the ones that
        # produced an item. `n_skipped` counts only seeds that ran out of tries,
        # and on the standard grid that number is 5 while this one is 1339.
        "n_rejected": sum(sum(c["build_rejections"].values()) for c in cells.values()),
        # Travels with the rows because rows get separated from manifests the
        # moment anyone loads the JSONL.
        "minimum_caveat": ("min_directives is minimal among the candidate directives "
                           "the generator produced, not proven globally minimal"),
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"__manifest__": manifest}) + "\n")
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    if verbose:
        print(f"wrote {len(rows)} items, {manifest['n_skipped']} seeds skipped, "
              f"{manifest['n_rejected']} candidates rejected -> {path}")
        print(f"taskset_hash {manifest['taskset_hash']}")
    return manifest


def read_manifest(path: str) -> Optional[Dict[str, Any]]:
    with open(path) as f:
        first = json.loads(f.readline())
    return first.get("__manifest__")
