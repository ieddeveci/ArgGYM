"""The six construction tasks own what a legal answer is; the evaluator owns delivery.

`docs/dataset-contract.md` section 1: which delimiters fence an answer comes from our
implementation, not from defeasible argumentation. So the fence is a render-time choice
-- the question names the template it was rendered with, another template renders another
sentence, the stored reference carries no fence at all, and an answer scores the same
bare. Section 5 adds the second half: `score_item` returns a `ScoreResult` whose
`success` is present on every branch, including the early zeros that never reach the goal
check.

One item per variant, not a grid: every property here is a function of a flag or of the
scorer, and `tests/e2e/` already sweeps the grid.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.core.answers import DEFAULT_TEMPLATE, AnswerTemplate, ScoreResult
from arggym.core.prompting import UNREADABLE, permitted_block
from arggym.core.scoring import score_item
from arggym.tasks import attack_defense as ad
from arggym.tasks import counter_argument as ca
from arggym.tasks import preference_construction as pc

# Any convention at all: the point is that the template is the caller's, not the
# dataset's. Square brackets are gone as a built-in because keeping a second
# accepted fence meant honouring one no prompt asks for.
OTHER_TEMPLATE = AnswerTemplate("boxed", "\\boxed{", "}")

LAST_LINK = "last_link_elitist"
DELIMITERS = ("<answer>", "</answer>")

# Level 3 is the cheapest cell that exists for all six, and each is a distinct code path:
# preference_construction and counter_argument write their own prompt, the three
# attack_defense modes share `core.prompting.render`.
# `make` takes the answer template, because which fence the question asks for is a
# render-time argument and one of the properties below is that changing it changes the
# question and nothing else.
VARIANTS = {
    "preference_construction": (lambda t: pc.make_item(3, 0, LAST_LINK, template=t),
                                lambda it: it.as_score_input()),
    "counter_argument": (lambda t: ca.make_item(3, 0, LAST_LINK, allow_strict=False,
                                                template=t),
                         ca.as_score_input),
    "counter_argument_strict": (lambda t: ca.make_item(3, 0, LAST_LINK, allow_strict=True,
                                                       template=t),
                                ca.as_score_input),
    "attack": (lambda t: ad.make_item(3, 0, LAST_LINK, mode="attack", template=t),
               lambda it: it.as_score_input()),
    "defence": (lambda t: ad.make_item(3, 0, LAST_LINK, mode="defence", template=t),
                lambda it: it.as_score_input()),
    "attack_defense": (lambda t: ad.make_item(3, 0, LAST_LINK, mode="attack_defense",
                                              template=t),
                       lambda it: it.as_score_input()),
}


@pytest.fixture(scope="module")
def items():
    out = {}
    for task, (make, as_input) in VARIANTS.items():
        it = make(DEFAULT_TEMPLATE)
        assert it is not None, f"{task}: no item at level 3 seed 0"
        out[task] = (it, as_input(it))
    return out


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_the_stored_reference_is_raw(task, items):
    it, _ = items[task]
    for d in DELIMITERS:
        assert d not in it.reference, f"{task}: reference still wraps its directives"


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_the_reference_scores_one_bare(task, items):
    it, score_input = items[task]
    r = score_item(it.reference, score_input)
    assert isinstance(r, ScoreResult)
    assert r.score == pytest.approx(1.0), (task, r.reason)
    assert r.success is True


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_the_reference_scores_one_fenced_after_reasoning(task, items):
    """What an evaluator actually sends: the answer after the model has thought aloud.

    This is the shape that scored zero before the fence came back, on every task.
    """
    it, score_input = items[task]
    r = score_item(f"Here is my reasoning.\n{DEFAULT_TEMPLATE.wrap(it.reference)}",
                   score_input)
    assert r.score == pytest.approx(1.0), (task, r.reason)
    assert r.success is True


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_the_prompt_names_the_fence_it_was_rendered_with(task, items):
    """The prompt says where to put the answer, and the template decides what it says.

    Both halves matter. Silence is what left a model guessing and a reasoning
    completion unscoreable; a hard-coded sentence is what made our delimiters a
    property of the benchmark instead of a property of one rendering.
    """
    it, _ = items[task]
    assert DEFAULT_TEMPLATE.instruction in it.prompt
    # The fence is named once, in the sentence the template contributed, and
    # nowhere in the clause that says what the answer must contain.
    assert it.prompt.count(DEFAULT_TEMPLATE.open) == 1


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_another_template_renders_another_sentence(task, items):
    """A harness with its own convention re-renders; it does not edit prompt strings."""
    make, _ = VARIANTS[task]
    other = make(OTHER_TEMPLATE)
    assert OTHER_TEMPLATE.instruction in other.prompt
    assert DEFAULT_TEMPLATE.instruction not in other.prompt
    # And nothing but that sentence moved.
    default, _ = items[task]
    assert (other.prompt.replace(OTHER_TEMPLATE.instruction, DEFAULT_TEMPLATE.instruction)
            == default.prompt)


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_the_prompt_still_states_what_a_legal_answer_is(task, items):
    """The delimiter phrase went; every clause that says what an answer *is* stayed."""
    it, _ = items[task]
    assert "Answer format: one directive per line." in it.prompt
    assert ("The answer must be minimal: one using more than twice the fewest directives "
            "that work scores zero.") in it.prompt
    if task == "preference_construction":
        # It writes its own prompt but is scored by `score_item` like the other five,
        # so the two rules that scorer applies unconditionally have to be stated here
        # too, and the forms it accepts have to be shown rather than left inferable
        # from the theory text (#21).
        assert "Permitted additions: preference directives only, written exactly in " \
               "these forms:" in it.prompt
        for form in ("   [prefer_rule: <rule> > <rule>]",
                     "   [prefer_premise: <literal> > <literal>]"):
            assert form in it.prompt
        assert UNREADABLE + ", so write only directives." in it.prompt
        assert pc.TIE_NOTE in it.prompt
        # And not the third rule: an answer of preferences alone cannot make a
        # consistent theory inconsistent, so stating it would describe a branch this
        # task cannot reach.
        assert "leaving the theory inconsistent" not in it.prompt
        return
    strict = "strict" in task
    block = permitted_block(strict)
    assert block in it.prompt, f"{task}: the permitted-directive block is not verbatim"
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
