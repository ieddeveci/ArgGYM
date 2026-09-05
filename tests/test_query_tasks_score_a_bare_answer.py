"""The six query tasks read the answer, not its wrapper, and say what they still require.

Each of these modules used to define its own answer-region regex and score 0.0
when it did not match, and each fused the delimiter phrase into the sentence that states
what the answer has to contain. `docs/dataset-contract.md` section 1 puts delivery on the
evaluator's side and content on ours, so the two are separate sentences now: the content
clause is the task's and never moves, and the sentence naming the fence comes from the
render-time template. A fenced answer and a bare one score the same.

Section 5 defines `success` per task. It is always returned, and it is not `score == 1.0`:
`formalization` succeeds on behavioural equivalence alone, so a correct theory written with
one spare directive is a success below 1.0.

One cheap cell per task. The grid is `tests/e2e`'s job.
"""
from __future__ import annotations

import pytest

from arggym.core.answers import DEFAULT_TEMPLATE, AnswerTemplate, ScoreResult
from arggym.tasks import (
    claim_chain,
    defeat_diagnosis,
    formalization,
    perturbation,
    semantics_query,
    status_query,
)

# Any convention at all: the point is that the template is the caller's, not the
# dataset's. Square brackets are gone as a built-in because keeping a second
# accepted fence meant honouring one no prompt asks for.
OTHER_TEMPLATE = AnswerTemplate("boxed", "\\boxed{", "}")

LEVEL, ORDERING, SEED = 3, "last_link_elitist", 0

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
        "   `defeated_at: <target>; defeater: <defeater>; kind: undermine|undercut|rebut`\n"],
    "formalization": ["Answer format: one directive per line."],
}


@pytest.fixture(scope="module")
def items():
    built = {}
    for task, module in MODULES.items():
        item = module.make_item(LEVEL, SEED, ORDERING)
        assert item is not None, f"{task}: no item at L{LEVEL} {ORDERING} seed {SEED}"
        built[task] = item
    return built


def reference_of(item) -> str:
    return item.reference() if callable(item.reference) else item.reference


@pytest.fixture
def item(request, items):
    return items[request.getfixturevalue("task")]


@pytest.mark.parametrize("task", sorted(MODULES))
def test_the_stored_reference_carries_no_delimiters(task, item):
    ref = reference_of(item)
    for d in ("<answer>", "</answer>"):
        assert d not in ref, f"{task}: the stored reference carries {d}"


@pytest.mark.parametrize("task", sorted(MODULES))
def test_the_bare_reference_scores_one(task, item):
    result = MODULES[task].score(reference_of(item), item)
    assert isinstance(result, ScoreResult)
    assert result.score == pytest.approx(1.0), result.reason
    assert result.success is True


@pytest.mark.parametrize("task", sorted(MODULES))
@pytest.mark.parametrize("open_tag, close_tag",
                         [("<answer>", "</answer>")])
def test_the_fenced_reference_scores_the_same(task, item, open_tag, close_tag):
    bare = MODULES[task].score(reference_of(item), item)
    fenced = MODULES[task].score(
        f"{open_tag}\n{reference_of(item)}\n{close_tag}", item)
    assert fenced.score == pytest.approx(bare.score)
    assert fenced.success is bare.success


@pytest.mark.parametrize("task", sorted(MODULES))
@pytest.mark.parametrize("open_tag, close_tag",
                         [("<answer>", "</answer>")])
def test_a_fence_around_reasoning_still_scores_one(task, item, open_tag, close_tag):
    """What an evaluator actually sends: the answer after the model's working.

    This is what the fence is for. Strict parsing reads everything inside it as an
    answer line, so without a fence the working itself would arrive as malformed
    answer and a reasoning model would score zero on every item.
    """
    text = (f"Let me work through it.\nFirst pass, wrong.\n"
            f"{open_tag}\n{reference_of(item)}\n{close_tag}")
    assert MODULES[task].score(text, item).score == pytest.approx(1.0)


@pytest.mark.parametrize("task", sorted(MODULES))
def test_the_prompt_names_the_fence_it_was_rendered_with(task, item):
    assert DEFAULT_TEMPLATE.instruction in item.prompt
    assert item.prompt.count(DEFAULT_TEMPLATE.open) == 1


@pytest.mark.parametrize("task", sorted(MODULES))
def test_another_template_moves_that_sentence_and_nothing_else(task, item):
    other = MODULES[task].make_item(LEVEL, SEED, ORDERING, template=OTHER_TEMPLATE)
    assert OTHER_TEMPLATE.instruction in other.prompt
    assert (other.prompt.replace(OTHER_TEMPLATE.instruction, DEFAULT_TEMPLATE.instruction)
            == item.prompt)


@pytest.mark.parametrize("task", sorted(MODULES))
def test_the_prompt_states_its_answer_content_rules(task, item):
    for clause in CONTENT_CLAUSES[task]:
        assert clause in item.prompt, f"{task}: lost `{clause}`"


def test_the_stable_sentence_survives_where_stable_is_asked():
    """The conditional clause is content, and level 3 is below the level that asks it."""
    level = next(l for l in (6, 9, 12, 15)
                 if semantics_query.STABLE in semantics_query.semantics_for(l))
    item = semantics_query.make_item(level, SEED, ORDERING)
    assert item is not None
    assert ("Under stable semantics, if the theory has no stable extension, answer "
            "`no stable extension`." in item.prompt)


@pytest.mark.parametrize("task", sorted(MODULES))
def test_an_empty_answer_is_scored_as_empty_not_as_undelivered(task, item):
    result = MODULES[task].score("", item)
    assert result.score == 0.0
    assert result.success is False
    assert result.reason != "no_answer_region"


# --- success is the task's own definition ---------------------------------------------


def flip(line: str, statuses=("justified", "overruled", "undecided")) -> str:
    head, _, status = line.rpartition(":")
    return f"{head}: {next(s for s in statuses if s != status.strip().lower())}"


@pytest.mark.parametrize("task", ["status_query", "semantics_query", "perturbation"])
def test_a_label_map_is_a_success_only_on_an_exact_match(task, item):
    lines = reference_of(item).splitlines()
    result = MODULES[task].score("\n".join([flip(lines[0])] + lines[1:]), item)
    assert result.success is False
    assert 0.0 < result.score < 1.0, "one wrong label, not a parse failure"


def test_claim_chain_needs_the_order_as_well_as_the_directives(items):
    item = items["claim_chain"]
    result = claim_chain.score("\n".join(reversed(item.reference.splitlines())), item)
    assert result.diagnostics["exact_match"] is True, "all and only the gold directives"
    assert result.diagnostics["correct_order"] is False
    assert result.success is False


def test_defeat_diagnosis_needs_the_status_as_well_as_the_failure_points(items):
    item = items["defeat_diagnosis"]
    lines = item.reference.splitlines()
    result = defeat_diagnosis.score("\n".join([flip(lines[0])] + lines[1:]), item)
    assert result.diagnostics["f1"] == pytest.approx(1.0), "every failure point is right"
    assert result.diagnostics["status_correct"] is False
    assert result.success is False


def test_formalization_succeeds_on_behaviour_alone(items):
    """A spare inert premise costs shape credit and changes no queried status."""
    item = items["formalization"]
    result = formalization.score(item.reference + "\n[premise: zzz9]", item)
    assert result.diagnostics["behavioural"] == pytest.approx(1.0)
    assert result.diagnostics["shape_f1"] < 1.0
    assert result.score < 1.0
    assert result.success is True


def test_formalization_fails_when_a_queried_status_moves(items):
    item = items["formalization"]
    lines = item.reference.splitlines()
    dropped = next(l for l in lines if l.startswith(("[defeasible", "[strict")))
    result = formalization.score("\n".join(l for l in lines if l != dropped), item)
    assert result.diagnostics["behavioural"] < 0.999
    assert result.success is False


def test_perturbation_does_not_offer_an_answer_no_item_can_have():
    """`none` scored zero on every item in the benchmark.

    `build` returns None when nothing changed, and the status-diversity guards under it
    would reject such a draw anyway, so gold is never empty. Inviting `none` offered a
    model an answer that is wrong by construction. The scorer still understands the
    word, because that is what it would mean if no-change items were ever generated.
    """
    it = perturbation.make_item(LEVEL, SEED, ORDERING)
    assert it.gold, "a shipped item with empty gold would put the sentence back"
    assert "none" not in it.prompt.rsplit("Answer format", 1)[1]
    assert perturbation.score("none", it).reason == "predicted_none"
