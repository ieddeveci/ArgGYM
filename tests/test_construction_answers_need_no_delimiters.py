"""The six construction tasks own what a legal answer is; the evaluator owns delivery.

`docs/dataset-contract.md` section 1: which delimiters fence an answer comes from our
implementation, not from defeasible argumentation. So no question names a fence at all,
the stored reference carries none either, and an answer scores the same bare. Section 5
adds the second half: `score_item` returns a `ScoreResult` whose `success` is present on
every branch, including the early zeros that never reach the goal check.

Every level of one cheap column, not level 3 alone (#110). What a question states and
whether its own reference still scores 1.0 bare are properties of an item, and a spec may
name any level the curriculum spans. `tests/e2e/` takes the ordering and seed axes and
covers the five exported levels only (`tests/e2e/conftest.py:22,36-38`), so the level
axis comes here.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.core.answers import DEFAULT_TEMPLATE, ScoreResult, extract_answer
from arggym.core.prompting import UNREADABLE, permitted_block, preference_block
from arggym.core.scoring import score_item
from arggym.tasks import attack_defense as ad
from arggym.tasks import counter_argument as ca
from arggym.tasks import preference_construction as pc

LAST_LINK = "last_link_elitist"
DELIMITERS = ("<answer>", "</answer>")

# Each variant is a distinct code path: preference_construction and counter_argument
# write their own prompt, the three attack_defense modes share `core.prompting.render`.
VARIANTS = {
    "preference_construction": (lambda lv: pc.make_item(lv, 0, LAST_LINK),
                                lambda it: it.as_score_input()),
    "counter_argument": (lambda lv: ca.make_item(lv, 0, LAST_LINK, allow_strict=False),
                         ca.as_score_input),
    "counter_argument_strict": (lambda lv: ca.make_item(lv, 0, LAST_LINK, allow_strict=True),
                                ca.as_score_input),
    "attack": (lambda lv: ad.make_item(lv, 0, LAST_LINK, mode="attack"),
               lambda it: it.as_score_input()),
    "defence": (lambda lv: ad.make_item(lv, 0, LAST_LINK, mode="defence"),
                lambda it: it.as_score_input()),
    "attack_defense": (lambda lv: ad.make_item(lv, 0, LAST_LINK, mode="attack_defense"),
                       lambda it: it.as_score_input()),
}

#: The curriculum range, not the five exported levels. `last_link_elitist` at seed 0 is
#: the cheap column; `weakest_link_democratic` is where the generation cost lives (#111).
LEVELS = tuple(range(1, 16))
CELLS = [(task, level) for task in sorted(VARIANTS) for level in LEVELS]

#: Five tests walk the same 90 cells, so one build serves all five. Under xdist they
#: scatter across workers and each rebuilds what it is handed; the cache is what keeps a
#: serial run of this file cheap.
_CACHE: dict = {}


def item_at(task: str, level: int):
    """The item and the mapping `score_item` takes, both cached."""
    if (task, level) not in _CACHE:
        make, as_input = VARIANTS[task]
        it = make(level)
        assert it is not None, f"{task}: no item at level {level} seed 0"
        _CACHE[task, level] = (it, as_input(it))
    return _CACHE[task, level]


@pytest.fixture
def cell(task, level):
    return item_at(task, level)


@pytest.mark.parametrize("task,level", CELLS)
def test_the_stored_reference_is_raw(task, cell):
    it, _ = cell
    for d in DELIMITERS:
        assert d not in it.reference, f"{task}: reference still wraps its directives"


@pytest.mark.parametrize("task,level", CELLS)
def test_the_reference_scores_one_bare(task, cell):
    it, score_input = cell
    r = score_item(it.reference, score_input)
    assert isinstance(r, ScoreResult)
    assert r.score == pytest.approx(1.0), (task, r.reason)
    assert r.success is True


@pytest.mark.parametrize("task,level", CELLS)
def test_a_completion_is_not_an_answer(task, cell):
    """What a model returns is not what the scorer takes.

    Turning the first into the second is the harness's job, and doing it here as
    well would mean honouring one convention above every other -- which is what
    stops a caller bringing their own.
    """
    it, score_input = cell
    completion = f"Here is my reasoning.\n{DEFAULT_TEMPLATE.wrap(it.reference)}"
    assert score_item(completion, score_input).score == 0.0
    r = score_item(extract_answer(completion), score_input)
    assert r.score == pytest.approx(1.0), (task, r.reason)
    assert r.success is True


@pytest.mark.parametrize("task,level", CELLS)
def test_the_prompt_says_nothing_about_where_to_put_the_answer(task, cell):
    """The question states the task. Where the answer goes is the harness's sentence,
    which is what lets one harness use XML tags and another a JSON schema."""
    it, _ = cell
    assert DEFAULT_TEMPLATE.instruction not in it.prompt
    assert DEFAULT_TEMPLATE.open not in it.prompt


# The directive forms, written out rather than read from `core.prompting`. Every block
# shares them through one constant now, so a test that imported it would pass on an empty
# one and take all seven variants down at once -- and no rejection reason is behind these
# lines, since a malformed preference is only `unparseable_lines`.
PREFERENCE_FORMS = ("   [prefer_rule: <rule> > <rule>]",
                    "   [prefer_premise: <literal> > <literal>]")
DEFEASIBLE_FORM = "   [defeasible <name>: <antecedent> => <consequent>]"
STRICT_FORM = "   [strict <name>: <antecedent> -> <consequent>]"
PREMISE_FORM = "   [premise: -<literal>]"


@pytest.mark.parametrize("task,level", CELLS)
def test_the_prompt_still_states_what_a_legal_answer_is(task, cell):
    """The delimiter phrase went; every clause that says what an answer *is* stayed."""
    it, _ = cell
    assert "Answer format: one directive per line." in it.prompt
    assert ("The answer must be minimal: one using more than twice the fewest directives "
            "that work scores zero.") in it.prompt
    for form in PREFERENCE_FORMS:
        assert form in it.prompt, f"{task}: lost the form line {form!r}"
    if task == "preference_construction":
        # It writes its own prompt but is scored by `score_item` like the other five,
        # so the two rules that scorer applies unconditionally have to be stated here
        # too, and the forms it accepts have to be shown rather than left inferable
        # from the theory text.
        block = preference_block()
        assert block and block in it.prompt, "the preference block is not verbatim"
        assert "Permitted additions: preference directives only, written exactly in " \
               "these forms:" in it.prompt
        assert UNREADABLE + ", so write only directives." in it.prompt
        assert pc.TIE_NOTE in it.prompt
        # And the forms it does not accept, since the header promises an exact list.
        for form in (DEFEASIBLE_FORM, STRICT_FORM, PREMISE_FORM):
            assert form not in it.prompt, f"preference_construction offers {form!r}"
        # And not the third rule: an answer of preferences alone cannot make a
        # consistent theory inconsistent, so stating it would describe a branch this
        # task cannot reach.
        assert "leaving the theory inconsistent" not in it.prompt
        return
    strict = "strict" in task
    block = permitted_block(strict)
    assert block and block in it.prompt, (
        f"{task}: the permitted-directive block is not verbatim")
    assert DEFEASIBLE_FORM in it.prompt, f"{task}: lost the defeasible form line"
    assert PREMISE_FORM in it.prompt, f"{task}: lost the premise form line"
    assert (STRICT_FORM in it.prompt) is strict, (
        f"{task}: the strict form is offered where the scorer rejects it")
    assert ("leaving the theory inconsistent" in it.prompt) is strict, (
        f"{task}: the consistency rule is stated where an answer cannot break it")
    for clause in ("Every rule needs a name, written after the kind and separated from it "
                   "by a space.",
                   "A name starts with a letter and continues with letters, digits or "
                   "underscores,",
                   "The arrow is => for a defeasible rule and -> for a strict one.",
                   "Several antecedents are joined with AND.",
                   "Rule antecedents must be literals already present in the theory.",
                   "A rule name in a consequent, written -<name>, switches that rule off.",
                   "New axioms may not be added.",
                   "A directive that cannot be read at all scores the whole answer zero."):
        assert clause in it.prompt, f"{task}: lost {clause!r}"


# a, d1 => p ; b, d2 => -p. p is justified once d1 outranks d2.
BASE = [
    Operation(kind="premise", content="a"),
    Operation(kind="premise", content="b"),
    Operation(kind="axiom", content="x"),
    Operation(kind="defeasible", name="d1", antecedents=("a",), consequent="p"),
    Operation(kind="defeasible", name="d2", antecedents=("b",), consequent="-p"),
]
GOLD = "[prefer_rule: d1 > d2]"


def _item(**kw):
    d = {"base_ops": BASE, "ordering": LAST_LINK,
         "goals": [{"claim": "p", "want": "JUSTIFIED"}], "min_directives": 1}
    d.update(kw)
    return d


@pytest.mark.parametrize("answer, reason", [
    ("", "no_directives"),
    ("I do not think anything can be done here.", "unparseable_lines:9"),
    (f"{GOLD}\nbanana", "unparseable_lines:1"),
    ("[axiom: z]", "all_directives_illegal"),
    ("[prefer_rule: d99 > d98]", "all_directives_illegal"),   # legal to write, engine rejects
    ("\n".join([GOLD] * 3), "bloated:3_used_vs_1_minimum"),
])
def test_success_is_false_on_every_early_zero(answer, reason):
    """None of these reached the goal check, so none of them met the goals.

    Returning no `success` here is what made `success_rate` an average over the rows that
    happened to reach a late branch (`docs/dataset-contract.md` section 5).
    """
    r = score_item(answer, _item())
    assert r.score == 0.0
    assert r.reason == reason
    assert r.success is False


def test_success_is_false_when_the_engine_rejects_the_theory(monkeypatch):
    """The last early-zero branch. No answer text reaches it on a generated item, so the
    engine is made to raise rather than left untested."""
    import arggym.core.scoring as scoring

    def boom(*a, **kw):
        raise RuntimeError("engine said no")

    monkeypatch.setattr(scoring, "build_framework", boom)
    r = scoring.score_item(GOLD, _item())
    assert r.score == 0.0
    assert r.reason == "engine_rejected:RuntimeError"
    assert r.success is False


def test_an_empty_answer_is_empty_rather_than_undelivered():
    """`no_answer_region` is gone: there is no missing-region outcome to report."""
    r = score_item("", _item())
    assert r.reason == "no_directives"
    assert "region" not in r.reason


def test_success_is_true_at_half_a_point_when_the_minimum_is_unknown():
    """Goal satisfaction is the task; economy is a second measurement on the same answer."""
    r = score_item(GOLD, _item(min_directives=None))
    assert r.score == 0.5
    assert r.reason == "success_but_minimum_unknown"
    assert r.success is True


def test_a_missed_goal_is_a_failure_that_still_carries_its_progress():
    r = score_item("[prefer_rule: d2 > d1]", _item())
    assert r.success is False
    assert r.reason == "goal_not_met"
    assert "progress" in r.diagnostics


@pytest.mark.parametrize("key", ["achieved_status", "deadlock_not_defeat", "efficiency"])
def test_the_task_specific_numbers_moved_into_diagnostics(key):
    """Four top-level fields, whatever the task, so a harness aggregates without knowing
    twelve return shapes."""
    r = score_item(GOLD, _item())
    assert set(r.as_dict()) == {"score", "success", "reason", "diagnostics"}
    assert key in r.diagnostics
