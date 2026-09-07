"""The six query tasks read the answer, not its wrapper, and say what they still require.

Each of these modules used to define its own answer-region regex and score 0.0
when it did not match, and each fused the delimiter phrase into the sentence that states
what the answer has to contain. `docs/dataset-contract.md` section 1 puts delivery on the
evaluator's side and content on ours, so the content clause is all the question carries
and the sentence naming a fence comes from the harness. A fenced answer and a bare one
score the same.

Section 5 defines `success` per task. It is always returned, and it is not `score == 1.0`:
`formalization` succeeds on behavioural equivalence alone, so a correct theory written with
one spare directive is a success below 1.0.

One cheap column per task, every level of it. What a question states and what its own
reference scores are properties of an item, and they were asserted at level 3 alone
(#110); `TasksetSpec` accepts any level the curriculum spans, so the column has to. The
ordering and seed axes stay with `tests/e2e`, which walks the two last-link/weakest-link
elitist orderings and both seeds -- and only the five exported levels
(`tests/e2e/conftest.py:22,36-38`), which is why the level axis comes here.
"""
from __future__ import annotations

import pytest

from arggym.core.answers import DEFAULT_TEMPLATE, ScoreResult, extract_answer
from arggym.tasks import (
    claim_chain,
    defeat_diagnosis,
    formalization,
    perturbation,
    semantics_query,
    status_query,
)

ORDERING, SEED = "last_link_elitist", 0
LEVELS = tuple(range(1, 16))

MODULES = {
    "status_query": status_query,
    "semantics_query": semantics_query,
    "perturbation": perturbation,
    "claim_chain": claim_chain,
    "defeat_diagnosis": defeat_diagnosis,
    "formalization": formalization,
}

# Every clause of the old answer-format text except the delimiter phrase. A paraphrase
# here is a change to the task: `claim_chain`'s scorer resolves each quoted line against
# the rendered theory, and `defeat_diagnosis`'s field names are its parse keys.
CONTENT_CLAUSES = {
    "status_query": ["Answer format: one line per claim, written as `claim: status`."],
    "semantics_query": [
        "Answer format: one line per query, written as `claim under semantics: status`."],
    "perturbation": [
        "Answer format: one line per changed claim, written as `claim: status`."],
    "claim_chain": [
        "Answer format: one directive per line, copied exactly as it appears above."],
    "defeat_diagnosis": [
        "Answer format:\n",
        "   first line: `status: overruled` or `status: undecided`\n",
        "   then one line per failure point, as\n",
        "   `defeated_at: <target>; defeater: <defeater>; kind: undermine|undercut|rebut`"],
    "formalization": ["Answer format: one directive per line."],
}


CELLS = [(task, level) for task in sorted(MODULES) for level in LEVELS]

#: Six tests walk all 90 cells and seven more walk one or three columns of them, so one
#: build serves thirteen. Under xdist they scatter across workers and each rebuilds what
#: it is handed; the cache is what keeps a serial run of this file cheap.
_CACHE: dict = {}


def item_at(task: str, level: int):
    if (task, level) not in _CACHE:
        it = MODULES[task].make_item(level, SEED, ORDERING)
        assert it is not None, f"{task}: no item at L{level} {ORDERING} seed {SEED}"
        _CACHE[task, level] = it
    return _CACHE[task, level]


def reference_of(item) -> str:
    return item.reference() if callable(item.reference) else item.reference


@pytest.fixture
def item(task, level):
    return item_at(task, level)


@pytest.mark.parametrize("task,level", CELLS)
def test_the_stored_reference_carries_no_delimiters(task, item):
    ref = reference_of(item)
    for d in ("<answer>", "</answer>"):
        assert d not in ref, f"{task}: the stored reference carries {d}"


@pytest.mark.parametrize("task,level", CELLS)
def test_the_bare_reference_scores_one(task, item):
    result = MODULES[task].score(reference_of(item), item)
    assert isinstance(result, ScoreResult)
    assert result.score == pytest.approx(1.0), result.reason
    assert result.success is True


@pytest.mark.parametrize("task,level", CELLS)
def test_a_completion_is_not_an_answer(task, item):
    """The scorer reads what it is handed, so a wrapper is unreadable answer text.

    Extracting here as well would mean honouring one convention above every other,
    which is exactly what stops a caller bringing their own. `extract_answer` is
    offered to a harness that wants the common one and called by nothing here.
    """
    fenced = f"Let me work through it.\n{DEFAULT_TEMPLATE.wrap(reference_of(item))}"
    assert MODULES[task].score(fenced, item).score == 0.0
    assert MODULES[task].score(extract_answer(fenced), item).score == pytest.approx(1.0)


@pytest.mark.parametrize("task,level", CELLS)
def test_the_prompt_says_nothing_about_where_to_put_the_answer(task, item):
    assert DEFAULT_TEMPLATE.instruction not in item.prompt
    assert DEFAULT_TEMPLATE.open not in item.prompt


@pytest.mark.parametrize("task,level", CELLS)
def test_the_prompt_states_its_answer_content_rules(task, item):
    for clause in CONTENT_CLAUSES[task]:
        assert clause in item.prompt, f"{task}: lost `{clause}`"


@pytest.mark.parametrize("level", LEVELS)
def test_the_stable_sentence_is_stated_exactly_where_stable_is_asked(level):
    """The conditional clause is content, so it belongs on the levels that schedule stable.

    Asserted in both directions over the range: a clause stated where nothing asks for it
    is noise a model has to read past, and this file is now the place that would notice
    `SEMANTICS_BY_LEVEL` and the prompt disagreeing.
    """
    sentence = ("Under stable semantics, if the theory has no stable extension, answer "
                "`no stable extension`.")
    asked = semantics_query.STABLE in semantics_query.semantics_for(level)
    assert (sentence in item_at("semantics_query", level).prompt) is asked


@pytest.mark.parametrize("task,level", CELLS)
def test_an_empty_answer_is_scored_as_empty_not_as_undelivered(task, item):
    result = MODULES[task].score("", item)
    assert result.score == 0.0
    assert result.success is False
    assert result.reason != "no_answer_region"


# --- success is the task's own definition ---------------------------------------------


def flip(line: str, statuses=("justified", "overruled", "undecided")) -> str:
    head, _, status = line.rpartition(":")
    return f"{head}: {next(s for s in statuses if s != status.strip().lower())}"


@pytest.mark.parametrize("task,level", [(t, lv) for t in
                                        ("status_query", "semantics_query", "perturbation")
                                        for lv in LEVELS])
def test_a_label_map_is_a_success_only_on_an_exact_match(task, item):
    lines = reference_of(item).splitlines()
    result = MODULES[task].score("\n".join([flip(lines[0])] + lines[1:]), item)
    assert result.success is False
    assert 0.0 < result.score < 1.0, "one wrong label, not a parse failure"


@pytest.mark.parametrize("level", LEVELS)
def test_claim_chain_needs_the_order_as_well_as_the_directives(level):
    item = item_at("claim_chain", level)
    result = claim_chain.score("\n".join(reversed(item.reference.splitlines())), item)
    assert result.diagnostics["exact_match"] is True, "all and only the gold directives"
    assert result.diagnostics["correct_order"] is False
    assert result.success is False


@pytest.mark.parametrize("level", LEVELS)
def test_defeat_diagnosis_needs_the_status_as_well_as_the_failure_points(level):
    item = item_at("defeat_diagnosis", level)
    lines = item.reference.splitlines()
    result = defeat_diagnosis.score("\n".join([flip(lines[0])] + lines[1:]), item)
    assert result.diagnostics["f1"] == pytest.approx(1.0), "every failure point is right"
    assert result.diagnostics["status_correct"] is False
    assert result.success is False


@pytest.mark.parametrize("level", LEVELS)
def test_formalization_succeeds_on_behaviour_alone(level):
    """A spare inert premise costs shape credit and changes no queried status."""
    item = item_at("formalization", level)
    result = formalization.score(item.reference + "\n[premise: zzz9]", item)
    assert result.diagnostics["behavioural"] == pytest.approx(1.0)
    assert result.diagnostics["shape_f1"] < 1.0
    assert result.score < 1.0
    assert result.success is True


def dropping_a_rule(item):
    """Each rule of the reference, dropped, with what the rest of it then scores.

    A rule the answer still parses without: dropping `q_14` out of a reference that also
    writes `[prefer_rule: q_14 > q_15]` leaves a preference naming nothing and the engine
    rejects the theory, which is a different outcome from a status that moved.
    """
    lines = item.reference.splitlines()
    for rule in (l for l in lines if l.startswith(("[defeasible", "[strict"))):
        result = formalization.score("\n".join(l for l in lines if l != rule), item)
        if result.diagnostics["behavioural"] is not None:
            yield rule, result


@pytest.mark.parametrize("level", [
    pytest.param(2, marks=pytest.mark.xfail(strict=True, reason=(
        "#121: at level 2 the item queries one claim and writes two routes to it, so "
        "every rule is on a redundant route and no removal moves a status"))),
    *[lv for lv in LEVELS if lv != 2],
])
def test_formalization_fails_when_a_queried_status_moves(level):
    """Which rule the queried statuses rest on is found rather than assumed.

    Dropping the first rule written was the probe, and it stops probing wherever the item
    concludes a queried claim more than one way -- at level 14 `ik7` has three routes, so
    the first of them can go and the answer is still behaviourally right. So the rule is
    chosen for its effect, and that a rule with an effect exists is asserted rather than
    relied on: an item without one scores `success` on an answer missing any rule.
    """
    item = item_at("formalization", level)
    moved = [(rule, r) for rule, r in dropping_a_rule(item)
             if r.diagnostics["behavioural"] < 0.999]
    assert moved, "no rule in the reference moves a queried status, so success cannot fail"
    for rule, result in moved:
        assert result.success is False, f"a moved status still succeeded without {rule}"


@pytest.mark.parametrize("level", LEVELS)
def test_perturbation_does_not_offer_an_answer_no_item_can_have(level):
    """`none` scored zero on every item in the benchmark.

    `build` rejects a draw where nothing changed, and the status-diversity guards under it
    would reject such a draw anyway, so gold is never empty. Inviting `none` offered a
    model an answer that is wrong by construction. The scorer still understands the
    word, because that is what it would mean if no-change items were ever generated.
    """
    it = item_at("perturbation", level)
    assert it.gold, "a shipped item with empty gold would put the sentence back"
    assert "none" not in it.prompt.rsplit("Answer format", 1)[1]
    assert perturbation.score("none", it).reason == "predicted_none"
