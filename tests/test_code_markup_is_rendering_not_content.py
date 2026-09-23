"""A code fence or backticks around an answer are rendering, and every parser drops them.

A chat model wraps anything that looks like code in a markdown fence or in backticks,
and the query prompts print their own answer format in backticks. Every parser already
ignored bullets and numbering; a fence line or a backtick was left over as a stray token
and zeroed a correct answer on all 12 tasks (#187). The fix lives in the parsers rather
than in a harness, so `arggym.score_row` scores a fenced answer the same for any caller.

What stays refused is a word: `Answer:`, `**Answer**`, or a stray word in backticks.
"""
from __future__ import annotations

import pytest

import arggym
from arggym.core.dataset import TaskDataset

TASKS = ("status_query", "semantics_query", "perturbation", "claim_chain",
         "defeat_diagnosis", "attack", "defence", "attack_defense", "counter_argument",
         "counter_argument_strict", "preference_construction", "formalization")
CELLS = [(t, lvl) for t in TASKS for lvl in (3, 9)]


def _lines(text):
    return text.splitlines()


MARKUP = {
    "fence": lambda r: "```\n" + r + "\n```",
    "fence_with_language": lambda r: "```text\n" + r + "\n```",
    "indented_fence": lambda r: "  ```dsl\n" + r + "\n  ```  ",
    "backticks_per_line": lambda r: "\n".join(f"`{x}`" for x in _lines(r)),
    "fence_and_backticks": lambda r: "```\n" + "\n".join(f"`{x}`" for x in _lines(r))
    + "\n```",
}
WORDS = {
    "answer_prefix": lambda r: "Answer:\n" + r,
    "bold_heading": lambda r: "**Answer**\n" + r,
    "backticked_stray_word": lambda r: r + "\n`therefore`",
    "fenced_stray_word": lambda r: "```\n" + r + "\ntherefore\n```",
}
# What the parsers forgave before #187, which must not move.
FORGIVEN = {
    "bullets": lambda r: "\n".join(f"- {x}" for x in _lines(r)),
    "numbered": lambda r: "\n".join(f"{i}. {x}" for i, x in enumerate(_lines(r), 1)),
    "crlf": lambda r: r.replace("\n", "\r\n"),
}


@pytest.fixture(scope="module", params=CELLS, ids=[f"{t}-L{lvl}" for t, lvl in CELLS])
def row(request):
    task, level = request.param
    entry, _ = TaskDataset(task, level, "last_link_elitist", size=1, seed=0).build_at(0)
    return entry


def _score(answer, entry):
    return arggym.score_row(answer, entry).score


@pytest.mark.parametrize("wrap", sorted(MARKUP))
def test_a_marked_up_reference_scores_what_the_bare_one_does(row, wrap):
    ref = row["reference_answer"]
    assert _score(ref, row) == 1.0
    assert _score(MARKUP[wrap](ref), row) == 1.0


@pytest.mark.parametrize("wrap", sorted(MARKUP))
def test_a_marked_up_partial_answer_scores_what_the_bare_one_does(row, wrap):
    """Not only the full-credit case: a fence must not move a partial score either."""
    first = _lines(row["reference_answer"])[0]
    got = arggym.score_row(MARKUP[wrap](first), row)
    bare = arggym.score_row(first, row)
    assert (got.score, got.reason) == (bare.score, bare.reason)


@pytest.mark.parametrize("wrap", sorted(WORDS))
def test_a_word_outside_the_lines_still_zeroes_the_answer(row, wrap):
    got = arggym.score_row(WORDS[wrap](row["reference_answer"]), row)
    assert got.score == 0.0
    assert got.reason.startswith("unparseable_tokens")


@pytest.mark.parametrize("wrap", sorted(FORGIVEN))
def test_what_was_forgiven_before_still_is(row, wrap):
    assert _score(FORGIVEN[wrap](row["reference_answer"]), row) == 1.0
