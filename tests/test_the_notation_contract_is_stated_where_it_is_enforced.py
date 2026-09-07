"""Every rule the scorer enforces is stated in the question and in `NOTATION.md`.

The contract is written three times: `core/prompting.py` states it to a model,
`NOTATION.md` states it to a reader, and `core/scoring.py` enforces it. #89 added one
clause to the naming grammar, it had to be written in two of those places by hand, and
nothing would have failed had the wrong subset moved. The table below is what fails.

The rejection reason is the key because it is the one name a rule already has. A clause
with no reason behind it is prose; a reason with no clause is a rule a model can only
find by losing.

What this does NOT catch:

- It checks that both places say *something* about each enforced rule, not that they say
  the same thing. "twice the minimum" here and "three times" there passes. Holding the
  two wordings identical is not wanted: they address different readers, and `NOTATION.md`
  writes "literals **ALREADY** present" where a prompt writes it plainly.
- It covers what `check_legality` and `score_item` reject. A rule the engine applies on
  its own -- an undercut aimed at a strict rule is inert -- is invisible here.
- The completeness assertion is a source scan for `reasons.append(f"...")` literals. A
  reason built in a loop, or assembled from a variable, slips past it -- and so does one
  returned through `failed(...)`, which is where three of the rows below come from
  (`bloated`, `unparseable_lines`, `inconsistent_theory`). Those three are hand-
  maintained, and a fourth added tomorrow fails nothing. Four more zeroing outcomes are
  outside the table altogether because no clause could state them: `no_directives`,
  `all_directives_illegal`, `engine_rejected` and `goal_not_met`.
- It reaches the six engine-scored tasks and no others. The six `MODULE`-scored tasks
  score their own answers, and each of them zeroes the whole answer on a single stray
  token -- `unparseable_tokens` on four of them and `unparseable_lines` on
  `formalization` and `perturbation`. `formalization` now says so, because its answer is
  the same DSL and `prompting.UNREADABLE` is already the right sentence for it; the other
  five answer in five other shapes and need five sentences of their own, which is #125.
  `claim_chain` is the one to read carefully there: its parser falls back to whole lines
  when the answer quotes nothing, so a fully prose answer parses and scores 0.0 with
  reason `ok` rather than raising. The rule it needs stated is still real -- prose mixed
  in *among* quoted directives does raise -- but its sentence cannot promise a stray word
  will be reported as unreadable.
"""
from __future__ import annotations

import pathlib
import re
from typing import Dict, NamedTuple, Optional, Set

import pytest

import arggym
from arggym.core import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEVEL, ORDERING = 3, "last_link_elitist"


class Clause(NamedTuple):
    """Where one rejection reason is stated, and whether the scorer appends it."""

    #: A phrase of the block the question carries. None where nothing states it.
    block: Optional[str]
    #: A phrase of `NOTATION.md`. None where nothing states it.
    notation: Optional[str]
    #: True where `scoring.py` builds this reason with `reasons.append`, which is what
    #: the completeness assertion below scans for.
    appended: bool = True


CLAUSES: Dict[str, Clause] = {
    "illegal_strict_rule": Clause(
        "Strict rules may not be added.", "no strict rules, except"),
    "illegal_non_preference": Clause(
        "No new rules or premises may be added.", "preference directives only"),
    "duplicate_rule_name": Clause(
        "no earlier line of the answer", "no earlier answer line"),
    "illegal_rule_name": Clause(
        "name starts with a letter and continues with letters, digits or underscores",
        "name starts with a letter and continues with letters, digits or underscores"),
    "illegal_name_collides_with_atom": Clause(
        "atom of the theory or of your answer", "atom of the theory or of the answer"),
    "illegal_unknown_antecedent": Clause(
        "antecedents must be literals already present in the theory",
        "antecedents must be literals already present in the theory"),
    "illegal_new_axiom": Clause("New axioms may not be added.", "no new axioms"),
    # One clause covers three reasons: a premise is legal only as the negation of an
    # ordinary premise, which rejects a bare assertion, an axiom's negation and a
    # contrary of nothing alike.
    "illegal_new_premise": Clause(
        "permitted only as the negation of an ordinary premise",
        "permitted only as the negation of an ordinary premise"),
    "illegal_undermine_axiom": Clause(
        "permitted only as the negation of an ordinary premise",
        "permitted only as the negation of an ordinary premise"),
    "illegal_asserted_contrary": Clause(
        "permitted only as the negation of an ordinary premise",
        "permitted only as the negation of an ordinary premise"),
    # The engine, not `check_legality`, is what refuses this one: a preference reaches
    # `build_framework` untouched and `add_rule_preference` / `add_premise_preference`
    # raise on an operand that is not a defeasible rule / not an ordinary premise
    # (`arggym/aspic/engine.py`). So a strict rule or an axiom the theory does show is
    # refused just as a name it does not show is.
    "rejected_preference": Clause(
        "A preference naming anything else is dropped",
        "A preference naming anything else is dropped"),
    "bloated": Clause(
        "more than twice the fewest directives that work scores zero",
        "more than twice the fewest directives that work scores zero", appended=False),
    "unparseable_lines": Clause(
        "directive that cannot be read at all scores the whole answer zero",
        "directive that cannot be read at all scores the whole answer zero",
        appended=False),
    "inconsistent_theory": Clause(
        "leaving the theory inconsistent", "leaving the theory inconsistent",
        appended=False),
}

#: Enforced and stated nowhere, on purpose, keyed by reason and holding why. Empty since
#: #102 stated the last of them. Kept rather than deleted: it is the only way to leave a
#: blank row in the table, and `test_the_gaps_are_exactly_the_rows_the_table_leaves_blank`
#: is what makes the next blank row cost a written reason.
ACCEPTED_GAPS: Dict[str, str] = {}

ENGINE_TASKS = sorted(n for n, s in registry.REGISTRY.items()
                      if s.scorer == registry.ENGINE)


def _flat(text: str) -> str:
    """Whitespace and case out. The two sites wrap and capitalize differently by design."""
    return " ".join(text.split()).lower()


#: Reachable on a preferences-only variant, and stated by another row's clause. A strict
#: rule there is rejected as `illegal_strict_rule`, because `scoring.py` checks the kind
#: a line before it checks `prefs_only` -- so the reason fires, but the sentence the
#: model needs is "No new rules or premises may be added." rather than the narrower one.
COVERED_BY = {"illegal_strict_rule": "illegal_non_preference"}


def _reachable(policy: Dict) -> Set[str]:
    """The reasons a variant's own answers can produce, from its scoring policy."""
    if policy.get("preferences_only"):
        # A rule, a premise or an axiom is dropped as soon as its kind is read, so no
        # naming, antecedent or premise check ever runs on one. Preferences alone cannot
        # break a consistent theory.
        return {"illegal_strict_rule", "illegal_non_preference", "bloated",
                "unparseable_lines", "rejected_preference"}
    out = set(CLAUSES) - {"illegal_non_preference", "inconsistent_theory"}
    if policy.get("allow_strict"):
        # Two contraries are both JUSTIFIED only if both are firm, and firmness needs a
        # strict rule, so this is the one variant whose answer can reach that branch.
        out = (out | {"inconsistent_theory"}) - {"illegal_strict_rule"}
    return out


def _phrase(reason: str, policy: Dict) -> Optional[str]:
    """The clause that has to reach a model of this variant for this reason."""
    if policy.get("preferences_only") and reason in COVERED_BY:
        return CLAUSES[COVERED_BY[reason]].block
    return CLAUSES[reason].block


@pytest.fixture(scope="module")
def questions() -> Dict[str, str]:
    """The rendered question, not the block: a block that stops shipping is #18."""
    return {task: arggym.TaskDataset(task, LEVEL, ORDERING, size=1)[0]["question"]
            for task in ENGINE_TASKS}


@pytest.mark.parametrize("task", ENGINE_TASKS)
def test_a_question_states_every_rule_its_own_answers_can_break(task, questions):
    q = _flat(questions[task])
    checked = 0
    policy = registry.get(task).policy
    for reason in sorted(_reachable(policy)):
        phrase = _phrase(reason, policy)
        if phrase is None:
            assert reason in ACCEPTED_GAPS, f"{reason} is stated nowhere and unrecorded"
            continue
        assert _flat(phrase) in q, f"{task}: nothing in the question states {reason}"
        checked += 1
    assert checked, f"{task}: the policy made every clause unreachable"


@pytest.mark.parametrize("task", ENGINE_TASKS)
def test_a_rule_a_question_cannot_reach_is_left_out_of_it(task, questions):
    """The other half of the table: a clause stated where it describes nothing is noise.

    Derived rather than listed, because a hand-picked list checks the cases somebody
    thought of. `preference_construction` is the case that matters -- it is scored by
    `score_item` like the other five, so the temptation is to give it their block, and
    every clause the other five carry is one this task's answers cannot produce.
    """
    policy = registry.get(task).policy
    q = _flat(questions[task])
    unreachable = set(CLAUSES) - _reachable(policy)
    checked = 0
    for reason in sorted(unreachable):
        phrase = CLAUSES[reason].block
        if phrase is None:
            continue
        # A phrase another reachable row also carries is not evidence of anything: the
        # three premise reasons share one clause.
        if any(CLAUSES[r].block == phrase for r in _reachable(policy)):
            continue
        checked += 1
        assert _flat(phrase) not in q, (
            f"{task} states {reason}, which its own answers cannot produce")
    assert checked, f"{task}: every clause is reachable, so this proves nothing"


def test_notation_md_states_every_rule_the_scorer_enforces():
    doc = _flat((ROOT / "NOTATION.md").read_text())
    for reason, clause in sorted(CLAUSES.items()):
        if clause.notation is None:
            assert reason in ACCEPTED_GAPS, f"{reason} is stated nowhere and unrecorded"
            continue
        assert _flat(clause.notation) in doc, f"NOTATION.md says nothing about {reason}"


_APPENDED = re.compile(r'reasons\.append\(f"([a-z_]+):')


def test_every_reason_the_scorer_appends_is_in_the_table():
    """The assertion that catches clause N+1 added and registered nowhere.

    Without it the table only covers the rules somebody remembered to add to it, which
    is the failure #38 names rather than a guard against it.
    """
    src = (ROOT / "arggym" / "core" / "scoring.py").read_text()
    found = set(_APPENDED.findall(src))
    assert found, "the scan matched nothing, so the reason strings changed shape"
    assert found == {r for r, c in CLAUSES.items() if c.appended}


def test_the_gaps_are_exactly_the_rows_the_table_leaves_blank():
    blank = {r for r, c in CLAUSES.items() if c.block is None or c.notation is None}
    assert blank == set(ACCEPTED_GAPS)
