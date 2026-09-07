"""The answer body is what the evaluator hands over, fenced or not.

`docs/dataset-contract.md` section 1: the fence is delivery, so no ArgGYM prompt
asks for one. A harness that calls a chat model still needs a fence, because a
strict parser has to know where the answer starts, and `AnswerTemplate` and
`extract_answer` are here for the common `<answer>` convention -- offered, never
required at scoring time. These tests pin that in every direction, and pin the
two behaviours that are not obvious: which region wins when there are several,
and which fence wins when both are present.
"""
import pytest

import arggym
from arggym.core import registry
from arggym.core.answers import (
    DEFAULT_TEMPLATE,
    XML_TAGS,
    AnswerTemplate,
    ScoreResult,
    extract_answer,
)

ORDERING = "last_link_elitist"

#: The curriculum range, not the five exported levels. Where a question ends is a
#: property of the rendered item and was asserted at level 3 alone (#110), while a spec
#: may name any level in this range and freeze it. No cache here: each cell is built by
#: one test and read once, so there is nothing to reuse.
LEVELS = tuple(range(1, 16))
CELLS = [(task, level) for task in sorted(registry.task_names()) for level in LEVELS]

#: The last thing each question says, which is the content clause `answer_format` was
#: handed. Naming the ending rather than a list of forbidden fences is what makes this
#: exhaustive: any sentence appended after the clause fails, whatever it says, not only
#: the five delimiters somebody thought of. `defeat_diagnosis` takes two, because its
#: format block gains a `survives_because` line on the items that have one.
#:
#: The six query tasks state these clauses again in `CONTENT_CLAUSES`
#: (`tests/test_query_tasks_score_a_bare_answer.py`). Both are asserted against the same
#: rendered question, so a reworded clause fails in both places rather than drifting.
TERMINAL_CLAUSE = {
    "attack": ("Answer format: one directive per line.",),
    "defence": ("Answer format: one directive per line.",),
    "attack_defense": ("Answer format: one directive per line.",),
    "counter_argument": ("Answer format: one directive per line.",),
    "counter_argument_strict": ("Answer format: one directive per line.",),
    "preference_construction": ("Answer format: one directive per line.",),
    "formalization": ("Answer format: one directive per line.",),
    "claim_chain": ("Answer format: one directive per line, copied exactly as it "
                    "appears above.",),
    "status_query": ("Answer format: one line per claim, written as `claim: status`.",),
    "semantics_query": ("Answer format: one line per query, written as "
                        "`claim under semantics: status`.",),
    "perturbation": ("Answer format: one line per changed claim, written as "
                     "`claim: status`.",),
    "defeat_diagnosis": ("`defeated_at: <target>; defeater: <defeater>; "
                         "kind: undermine|undercut|rebut`",
                         "...; survives_because: <rule>"),
}


def test_the_ending_table_covers_the_registry():
    """A task added without an entry would otherwise be checked by nothing."""
    assert set(TERMINAL_CLAUSE) == set(registry.task_names())


@pytest.mark.parametrize("task,level", CELLS)
def test_no_task_renders_a_delivery_sentence(task, level):
    """Generation has no way to say where the answer goes, on any of the twelve.

    It used to: every `build` took a template and threaded it into `answer_format`, so
    a frozen taskset could carry the sentence the harness appends and a model would be
    told twice. One test over the registry rather than two per-task copies, because the
    rule is the dataset's and holds whatever the task is.

    The question has to *end* with its content clause. Checking a list of fence strings
    instead let "then put your response inside `<solution>` tags" through, which is the
    same defect in a wording nobody enumerated.
    """
    q = arggym.TaskDataset(task, level, ORDERING, size=1)[0]["question"]
    assert q.rstrip().endswith(TERMINAL_CLAUSE[task]), (
        f"{task} L{level}: something follows the answer-format clause")
    # Named separately because it is the exact sentence #53 was about, and because a
    # fence could in principle arrive somewhere other than the end.
    assert DEFAULT_TEMPLATE.instruction not in q
    assert DEFAULT_TEMPLATE.open not in q


def test_a_bare_answer_is_its_own_body():
    assert extract_answer("[prefer_rule: r1 > r2]") == "[prefer_rule: r1 > r2]"


def test_a_fenced_answer_gives_up_its_body():
    assert extract_answer("thinking\n<answer>\n[prefer_rule: r1 > r2]\n</answer>").strip() \
        == "[prefer_rule: r1 > r2]"


def test_only_the_fence_the_question_asks_for_is_read():
    # A second accepted convention would mean the scorer honours something no
    # prompt requests -- the stated-versus-enforced mismatch this work keeps
    # closing -- and would need a precedence rule for answers carrying both.
    # Square brackets are directive syntax here, so the text below is an answer
    # whose first and last lines are unreadable, not a fenced one.
    text = "[answer]\n[prefer_rule: a > b]\n[/answer]"
    assert extract_answer(text) == text


def test_the_default_template_is_the_reasoning_gym_one():
    # reasoning-gym's system prompts ask for <answer>answer here</answer> and its
    # `utils.extract_answer` reads the region back with the same tag name.
    assert DEFAULT_TEMPLATE is XML_TAGS
    assert (XML_TAGS.open, XML_TAGS.close) == ("<answer>", "</answer>")
    assert XML_TAGS.instruction == "Give your final answer between <answer> and </answer>."


def test_a_template_is_data_so_a_harness_can_bring_its_own():
    mine = AnswerTemplate("boxed", "\\boxed{", "}")
    assert mine.instruction == "Give your final answer between \\boxed{ and }."
    assert mine.wrap("x") == "\\boxed{\nx\n}"


def test_the_last_region_wins_because_the_first_is_the_draft():
    # A reasoning model drafts a candidate mid-thought and then revises it. The
    # first region is the draft, so taking it scores work the model discarded.
    text = "<answer>\n[prefer_rule: a > b]\n</answer>\nwait, that undermines c.\n" \
           "<answer>\n[prefer_rule: b > a]\n</answer>"
    assert extract_answer(text).strip() == "[prefer_rule: b > a]"


def test_nothing_at_all_is_an_empty_body_not_an_error():
    assert extract_answer(None) == ""
    assert extract_answer("") == ""


def test_an_unclosed_region_is_not_a_region():
    # Half a wrapper is a truncated generation, not a delimiter contract. The
    # text is returned whole so the parser sees what the model actually wrote.
    assert extract_answer("<answer>\n[prefer_rule: a > b]") \
        == "<answer>\n[prefer_rule: a > b]"


def test_success_is_carried_separately_from_the_score():
    # The contract's reason for a result object: a construction answer that
    # meets every goal is a success at 0.5 when the minimum is unknown.
    r = ScoreResult(score=0.5, success=True, reason="success_but_minimum_unknown")
    assert r.success and r.score < 1.0
    assert r.as_dict()["success"] is True
