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
ordering and seed axes stay with `tests/e2e`, which walks the two elitist orderings
and both seeds over the five exported levels -- and under `make test` only levels 3,
6 and 9 at seed 0, since everything outside `FAST_GRID` is marked slow and
`addopts` deselects it (`tests/e2e/conftest.py:22,36-38`). So no default run reaches
a level off the grid, which is why the level axis comes here.
"""
from __future__ import annotations

import re

import pytest

from arggym.core import prompting
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
#: The four the curriculum ships. `ORDERING` above is the one `item_at` builds at,
#: so anything rendered conditionally on the ordering needs these as well.
ORDERINGS = ("last_link_elitist", "last_link_democratic",
             "weakest_link_elitist", "weakest_link_democratic")
#: One level for the ordering row. Every task builds here on all four.
BLOCK_LEVEL = 3

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

#: Six tests walk all 90 cells and seven more walk one or three columns of them, so a
#: cell is read six to eight times depending on its task. Under xdist they scatter
#: across workers and each rebuilds what it is handed; the cache is what keeps a
#: serial run of this file cheap.
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

    Both directions, because a clause stated where nothing asks for it is noise a model
    has to read past. `test_semantics_query_is_not_status_query.py` already asserts this
    pair on the exported grid, so what this adds is the ten off-grid levels, on a build
    the tests above have already paid for.
    """
    sentence = ("Under stable semantics, if the theory has no stable extension, answer "
                "`no stable extension`.")
    asked = semantics_query.STABLE in semantics_query.semantics_for(level)
    assert (sentence in item_at("semantics_query", level).prompt) is asked


#: The five that carry `prompting.STRAY_TEXT`. `formalization` is the sixth `MODULE`
#: task and states the same all-or-nothing rule through `prompting.UNREADABLE`, in the
#: notation block #126 gave it: its answer is the DSL, so "a directive that cannot be
#: read" is the right noun there and the shared sentence would state the rule twice.
STRAY_TASKS = tuple(t for t in sorted(MODULES) if t != "formalization")

#: The two sentences, written out rather than imported. `x in prompt` is true of the
#: empty string, so a guard that only imported the constant would still pass with that
#: constant emptied -- the hole #126's first guard had. Written out, the constant has to
#: equal this, so a rewording fails rather than drifting.
STRAY_SENTENCE = "Any word in your answer outside these lines scores the whole answer zero."
KIND_SENTENCE = ("The kind must be written as one of those three words, in any "
                 "capitalisation; any other scores the whole answer zero.")

#: The whole last paragraph of each question, which is the answer-format block and
#: nothing else -- verified over all 15 levels and all four orderings, where every one of
#: the five renders exactly the block below and `defeat_diagnosis` one of its two.
#:
#: Pinning the whole block rather than its ending is what `TERMINAL_CLAUSE`
#: (`tests/test_delimiters_are_delivery_not_content.py`) cannot do. `endswith` sees the
#: tail, so a sentence inserted between the clause and the shared one passes there, and
#: so does `answer_format` growing a line of its own above the clause. Equality over the
#: paragraph sees both.
ANSWER_BLOCK = {
    "status_query": (
        "Answer format: one line per claim, written as `claim: status`.\n"
        + STRAY_SENTENCE,),
    "semantics_query": (
        "Answer format: one line per query, written as `claim under semantics: status`.\n"
        + STRAY_SENTENCE,),
    "perturbation": (
        "Answer format: one line per changed claim, written as `claim: status`.\n"
        + STRAY_SENTENCE,),
    "claim_chain": (
        "Answer format: one directive per line, copied exactly as it appears above.\n"
        + STRAY_SENTENCE,),
    # Keyed by `requires_survival_reason`, so the variant is checked against the item's
    # own record of whether it asks for one rather than against a list that accepts
    # either. A build that stopped rendering the template line would otherwise match the
    # False entry on every item and fail nothing.
    "defeat_diagnosis": {
        False: ("Answer format:\n"
                "   first line: `status: overruled` or `status: undecided`\n"
                "   then one line per failure point, as\n"
                "   `defeated_at: <target>; defeater: <defeater>; "
                "kind: undermine|undercut|rebut`\n"
                + KIND_SENTENCE + "\n" + STRAY_SENTENCE),
        True: ("Answer format:\n"
               "   first line: `status: overruled` or `status: undecided`\n"
               "   then one line per failure point, as\n"
               "   `defeated_at: <target>; defeater: <defeater>; "
               "kind: undermine|undercut|rebut`\n"
               "   ...; survives_because: <rule>\n"
               + KIND_SENTENCE + "\n" + STRAY_SENTENCE),
    },
}


def expected_block(task: str, item) -> str:
    if task == "defeat_diagnosis":
        return ANSWER_BLOCK[task][item.metadata["requires_survival_reason"]]
    return ANSWER_BLOCK[task][0]


def rendered_block(item) -> str:
    """The question's last blank-line-separated paragraph."""
    return item.prompt.split("\n\n")[-1]


@pytest.mark.parametrize("task,level",
                         [(t, lv) for t in STRAY_TASKS for lv in LEVELS])
def test_the_question_ends_with_exactly_its_answer_format_block(task, level):
    """Every level of each column, against the block written out here.

    What this does NOT catch: a sentence added to the question somewhere above this
    paragraph. The theory text is in there, so no table can pin the whole question, and
    a body sentence reading "you may add a short explanation" would leave every
    assertion in this file green while contradicting the block below it. Reviewing a
    change to a `_render_prompt` is what covers that.
    """
    item = item_at(task, level)
    assert rendered_block(item) == expected_block(task, item)


@pytest.mark.parametrize("task", STRAY_TASKS)
def test_the_answer_format_block_does_not_vary_with_the_ordering(task):
    """The axis the column above holds fixed, and the one a conditional would hide in.

    `item_at` builds at `last_link_elitist` alone, so a block rendered only under that
    ordering keeps 30 of the 40 shipped rows of a task and passes every other assertion
    here. One level over the four orderings is enough to see it, since the level axis is
    already walked above.
    """
    for ordering in ORDERINGS:
        item = MODULES[task].make_item(BLOCK_LEVEL, SEED, ordering)
        assert item is not None, f"{task}: no item at L{BLOCK_LEVEL} {ordering}"
        assert rendered_block(item) == expected_block(task, item), (
            f"{task}: the block moved under {ordering}")


def test_both_defeat_diagnosis_blocks_are_reached():
    """Neither entry of the table above is dead, so neither passes by never being used."""
    seen = {item_at("defeat_diagnosis", lv).metadata["requires_survival_reason"]
            for lv in LEVELS}
    assert seen == {False, True}


@pytest.mark.parametrize("task,level",
                         [(t, lv) for t in STRAY_TASKS for lv in LEVELS])
def test_the_prompt_states_that_a_stray_word_zeroes_the_answer(task, level):
    """The rule these five enforce and none of them used to state (#125).

    Behavioural as well as textual. All five refuse a stray word out of the parser --
    `unparseable_tokens:1` on four and `unparseable_tokens:1` on `perturbation` -- and it
    is that refusal the second half guards: the junk check still zeroes a word the format
    clause did not license. A wording check alone would survive the junk check going
    away.

    It does not guard `claim_chain`'s bracketing, which is what makes a *prose-only*
    answer worth nothing there without raising. Stripping the brackets off both gold and
    the parsed lines leaves this test green and fails
    `tests/test_an_answer_is_a_value_not_its_text.py` instead.

    One word, not one line: each of the five discards a bullet or a bare numeral before
    it counts, so a bulleted or numbered reference still scores 1.0 while `therefore`
    on a line of its own scores 0.0.
    """
    item = item_at(task, level)
    assert prompting.STRAY_TEXT == STRAY_SENTENCE
    assert STRAY_SENTENCE in item.prompt, f"{task}: lost the stray-word sentence"
    ref = reference_of(item)
    assert MODULES[task].score(ref, item).score == pytest.approx(1.0)
    assert MODULES[task].score(ref + "\ntherefore", item).score == 0.0


@pytest.mark.parametrize("level", LEVELS)
def test_defeat_diagnosis_states_that_only_three_kind_words_are_read(level):
    """Its second whole-answer rule, which `STRAY_SENTENCE` does not cover.

    The bad word sits inside a well-shaped line rather than outside one, so nothing
    about stray text reaches it: `parse` raises `invalid_kind` on any `kind:` word
    outside the three and the answer scores 0.0 before a record is compared. The
    capitalisation half of the sentence is asserted too, in the direction that costs a
    model something: a shipped reference with every kind upper-cased still scores 1.0.
    """
    item = item_at("defeat_diagnosis", level)
    assert defeat_diagnosis.KIND_RULE == KIND_SENTENCE
    assert KIND_SENTENCE in item.prompt, "lost the kind sentence"
    ref = reference_of(item)
    kinds = re.compile(r"kind:\s*(undermine|undercut|rebut)\b")
    assert kinds.search(ref), "the reference names no kind, so this proves nothing"
    assert defeat_diagnosis.score(ref, item).score == pytest.approx(1.0)

    result = defeat_diagnosis.score(kinds.sub("kind: rebuttal", ref), item)
    assert result.score == 0.0
    assert result.reason.startswith("invalid_kind")

    shouted = kinds.sub(lambda m: "kind: " + m.group(1).upper(), ref)
    assert defeat_diagnosis.score(shouted, item).score == pytest.approx(1.0)


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


@pytest.mark.parametrize("level", LEVELS)
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
