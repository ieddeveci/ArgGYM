"""A prompt has to say which directives its own scorer will accept.

`check_legality` rejects a strict rule unless the item allows one. `counter_argument`
rendered both variants byte-identically, so the only way to tell an item that accepts a
one-directive strict answer from one that scores the same answer zero was to guess (#37).
The block that lists the permitted forms existed but was gated off, so it reached no
item at all (#38, #18).
"""
from __future__ import annotations

import inspect

import pytest

from arggym.core.curriculum import ATTACK, DEFENCE, MIXED
from arggym.core.export import ALL_ORDERINGS, export_task
from arggym.core.prompting import permitted_block
from arggym.core.scoring import score_item
from arggym.tasks import attack_defense as ad
from arggym.tasks import counter_argument as ca
from arggym.tasks import preference_construction as pc

GRID = inspect.signature(export_task).parameters["levels"].default
SEEDS = inspect.signature(export_task).parameters["seeds"].default
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]
# The block is the same on every item of a variant, so sweeping the whole grid only
# re-times generation. The cheap levels carry the default run; the sweep is marked slow.
CHEAP = [(lv, o, s) for lv in (3, 6) for o in ALL_ORDERINGS for s in SEEDS]
# Below level 8 `contested` is already True only for the strict arm, so the two variants
# built different theories and the prompts differed for a reason that has nothing to do
# with this change. Byte-identity, the defect in #37, held only from level 8 up.
IDENTICAL_BEFORE = [(lv, o, s) for lv in GRID if lv >= 8 for o in ALL_ORDERINGS for s in SEEDS]
CHEAPEST_IDENTICAL = [c for c in IDENTICAL_BEFORE if c[0] == 9]

HEADER = "Permitted additions"
NO_STRICT = "Strict rules may not be added."
STRICT_FORM = "[strict <name>: <antecedent> -> <consequent>]"


@pytest.mark.parametrize("level,ordering,seed", CHEAPEST_IDENTICAL)
def test_the_two_counter_argument_variants_no_longer_read_alike(level, ordering, seed):
    """The cells where the two prompts used to be byte-identical."""
    plain = ca.make_item(level, seed, ordering, allow_strict=False)
    strict = ca.make_item(level, seed, ordering, allow_strict=True)
    assert plain is not None and strict is not None
    assert plain.theory_text == strict.theory_text, (
        "this cell no longer shares a theory between the variants, so it does not test "
        "what it was chosen for")
    assert plain.prompt != strict.prompt, "the ablation is invisible to the model"


@pytest.mark.parametrize("allow_strict", [False, True])
@pytest.mark.parametrize("level,ordering,seed", CHEAP)
def test_each_counter_argument_prompt_states_its_own_strict_policy(
        level, ordering, seed, allow_strict):
    it = ca.make_item(level, seed, ordering, allow_strict=allow_strict)
    assert it is not None
    assert HEADER in it.prompt
    assert it.metadata["allow_strict"] is allow_strict
    if allow_strict:
        assert STRICT_FORM in it.prompt and NO_STRICT not in it.prompt
    else:
        assert NO_STRICT in it.prompt and STRICT_FORM not in it.prompt


@pytest.mark.parametrize("mode", [ATTACK, DEFENCE, MIXED])
def test_attack_defense_ships_the_block_it_already_had(mode):
    seen = 0
    for lv, o, s in CHEAP:
        it = ad.make_item(lv, s, o, mode=mode)
        if it is None:
            continue
        seen += 1
        assert HEADER in it.prompt, f"L{lv} {o} seed {s} {mode}"
        assert NO_STRICT in it.prompt
    assert seen, f"no {mode} item generated"


def test_the_strict_answer_is_permitted_exactly_where_the_prompt_says_it_is():
    """The behaviour behind the wording, not the wording alone.

    Only the cells whose strict item accepts the one-directive answer: the shortcut is
    offered to `bank` and taken only when it reaches both goals, so `min_directives == 1`
    is the test for whether this cell has anything to say.
    """
    checked = 0
    for level, ordering, seed in CHEAPEST_IDENTICAL:
        strict = ca.make_item(level, seed, ordering, allow_strict=True)
        plain = ca.make_item(level, seed, ordering, allow_strict=False)
        assert strict is not None and plain is not None
        if strict.min_directives != 1:
            continue
        checked += 1
        answer = f"[strict cs: {strict.seed_lit} -> -{strict.target}]"

        accepted = score_item(answer, ca.as_score_input(strict))
        assert accepted.score == pytest.approx(1.0), (level, ordering, seed, accepted)
        assert STRICT_FORM in strict.prompt

        rejected = score_item(answer, ca.as_score_input(plain))
        assert rejected.score == 0.0
        assert "illegal" in rejected.reason, rejected.reason
        assert NO_STRICT in plain.prompt
    assert checked, "no cell at this level takes the strict shortcut; pick another level"


def test_the_block_is_a_function_of_its_flag_and_nothing_else():
    """Why the grid sweeps this file used to carry are gone.

    `permitted_block` takes one boolean and its output is inserted verbatim, so one item
    per variant settles it for every item. Sweeping levels 12 and 15 to re-read the same
    two strings cost the slow tier about forty minutes and proved nothing the two lines
    below do not. That every reference still scores 1.0 across the whole grid is checked
    by `tests/e2e/test_generation.py::test_reference_scores_one`, which already runs there.
    """
    assert permitted_block(False) == permitted_block(False)
    blocks = {allow: {ca.make_item(lv, s, o, allow_strict=allow).prompt.split(HEADER)[1]
                      for lv, o, s in CHEAP for _ in (0,)}
              for allow in (False, True)}
    for allow, seen in blocks.items():
        assert len(seen) == 1, f"the block varies between items at allow_strict={allow}"


@pytest.mark.parametrize("allow_strict", [False, True])
def test_the_block_states_the_rules_that_zero_an_answer(allow_strict):
    """Each of these is enforced and was unsaid, so a model could only find it by losing."""
    block = permitted_block(allow_strict)
    assert "joined with AND" in block, "the parser accepts several antecedents"
    assert "no earlier line of the answer, has used" in block, "duplicate names score 0"
    assert "more than twice" in block, "the bloat factor scores 0"
    assert "without a leading -" in block, "only a positively written premise can be negated"
    assert "cannot be read at all" in block, "one unparseable line zeroes the answer"
    # Consistency is stated where an answer can break it and nowhere else. Two contraries
    # are both JUSTIFIED only if both are firm, firmness needs an axiom and a strict rule,
    # and no answer may add an axiom -- so only a strict variant can reach the branch.
    assert ("leaving the theory inconsistent" in block) is allow_strict


def test_preference_construction_states_the_rule_that_zeroes_its_answer():
    """It sets `min_directives` and so hits the same bloat factor, and never said so."""
    it = pc.make_item(9, 0, "last_link_elitist")
    assert it is not None and it.min_directives
    assert "more than twice" in it.prompt
