"""A prompt has to say which directives its own scorer will accept.

`check_legality` rejects a strict rule unless the item allows one. `counter_argument`
rendered both variants byte-identically, so the only way to tell an item that accepts a
one-directive strict answer from one that scores the same answer zero was to guess (#37).
The block that lists the permitted forms existed but was gated off, so it reached no
item at all (#38, #18).
"""
from __future__ import annotations

import pytest

from arggym.core.curriculum import ATTACK, DEFENCE, MIXED
from arggym.core.prompting import (
    PREFERENCE_DROPPED,
    PREFERENCE_OPERANDS,
    UNREADABLE,
    formalization_notation,
    permitted_block,
    preference_block,
)
from arggym.core.scoring import score_item
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import attack_defense as ad
from arggym.tasks import counter_argument as ca
from arggym.tasks import formalization as fm
from arggym.tasks import preference_construction as pc

GRID = LEVELS
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]
# The block is the same on every item of a variant, so sweeping the whole grid only
# re-times generation. The cheap levels carry the default run; the sweep is marked slow.
CHEAP = [(lv, o, s) for lv in (3, 6) for o in ALL_ORDERINGS for s in SEEDS]
# Byte-identity, the defect in #37, held only from level 8 up when this was written:
# below it `contested` was True for the strict arm alone, so the two variants built
# different theories and their prompts differed for a reason that has nothing to do with
# this change. The theories are shared from level 6 up now, and at level 3 wherever the
# entry-level draw leaves `n_strict` alone; level 9 stays the cell these cases use.
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

    Permitted is about legality, not about winning. The check used to score the strict
    one-liner 1.0 on the strict arm, which read as one assertion but was two: the scorer
    accepts a strict rule here, and this line is a whole answer. The second half was #93's
    defect and it is gone, so the cells it needed -- `min_directives == 1` at level 9 --
    are gone with it and the skip that hunted for them went too. Legality is a property of
    every cell, so all eight are checked now instead of the few that took the shortcut.
    """
    for level, ordering, seed in CHEAPEST_IDENTICAL:
        strict = ca.make_item(level, seed, ordering, allow_strict=True)
        plain = ca.make_item(level, seed, ordering, allow_strict=False)
        assert strict is not None and plain is not None
        where = (level, ordering, seed)
        answer = f"[strict cs: {strict.seed_lit} -> -{strict.target}]"

        accepted = score_item(answer, ca.as_score_input(strict))
        assert accepted.diagnostics["illegal"] == [], (where, accepted)
        assert STRICT_FORM in strict.prompt

        rejected = score_item(answer, ca.as_score_input(plain))
        assert rejected.score == 0.0, (where, rejected)
        assert "illegal" in rejected.reason, (where, rejected.reason)
        assert NO_STRICT in plain.prompt


def test_a_block_is_a_function_of_its_variant_and_nothing_else():
    """Why the grid sweeps this file used to carry are gone.

    Four blocks cover the seven variants that carry one: `permitted_block` takes a
    boolean, `preference_block` and `formalization_notation` take nothing, and each
    output is inserted verbatim. So one item per variant settles it for every item.
    Sweeping levels 12 and 15 to re-read the same strings cost the slow tier about forty
    minutes and proved nothing the lines below do not. That every reference still scores
    1.0 across the whole grid is checked by
    `tests/e2e/test_generation.py::test_reference_scores_one`, which already runs there.
    """
    assert permitted_block(False) == permitted_block(False)
    blocks = {allow: {ca.make_item(lv, s, o, allow_strict=allow).prompt.split(HEADER)[1]
                      for lv, o, s in CHEAP}
              for allow in (False, True)}
    for allow, seen in blocks.items():
        assert len(seen) == 1, f"the block varies between items at allow_strict={allow}"
    for name, block, mod in (("preference_construction", preference_block(), pc),
                             ("formalization", formalization_notation(), fm)):
        # `block and`, because `"" in prompt` is true: an emptied block would otherwise
        # delete the whole DSL from the question and pass every assertion here.
        assert block, f"{name}: the block is empty"
        built = 0
        for lv, o, s in CHEAP:
            it = mod.make_item(lv, s, o)
            if it is None:
                continue
            built += 1
            assert block in it.prompt, f"{name} L{lv} {o} seed {s}: block not verbatim"
        assert built, f"no {name} item generated"


@pytest.mark.parametrize("allow_strict", [False, True])
def test_the_block_states_the_rules_that_zero_an_answer(allow_strict):
    """Each of these is enforced and was unsaid, so a model could only find it by losing."""
    block = permitted_block(allow_strict)
    assert "joined with AND" in block, "the parser accepts several antecedents"
    assert "no earlier line of the answer" in block, "duplicate names score 0"
    assert "atom of the theory or of your answer" in block, \
        "a name shared with an atom scores 0, including an atom the answer introduces (#89)"
    assert "more than twice" in block, "the bloat factor scores 0"
    assert "without a leading -" in block, "only a positively written premise can be negated"
    assert "cannot be read at all" in block, "one unparseable line zeroes the answer"
    # Consistency is stated where an answer can break it and nowhere else. Two contraries
    # are both JUSTIFIED only if both are firm, firmness needs an axiom and a strict rule,
    # and no answer may add an axiom -- so only a strict variant can reach the branch.
    assert ("leaving the theory inconsistent" in block) is allow_strict


def test_formalization_states_the_two_rules_that_zero_its_answer():
    """The half of #102 that no other guard reaches.

    `formalization` is `MODULE`-scored, so the clause table in
    `tests/test_the_notation_contract_is_stated_where_it_is_enforced.py` skips it, and
    `test_a_block_is_a_function_of_its_variant_and_nothing_else` above asserts the block
    ships verbatim whatever it says. Deleting either sentence used to fail nothing.

    Both cost the whole answer, not a directive. `parse` raises on any line the DSL cannot
    read -- the shipped reference inside a markdown fence scores 0.0 with
    `unparseable_tokens:2` -- and `status_map` swallows the engine's refusal of a
    preference into an empty map that `score_value` turns into `engine_rejected`, so a
    `prefer_rule` over the answer's own strict rule, or a `prefer_premise` over its own
    axiom, scores 0.0 where the same directive costs one line of economy on the six
    engine tasks (`arggym/tasks/formalization.py`).
    """
    block = formalization_notation()
    assert UNREADABLE in block, "one unreadable line zeroes the answer"
    # The operand rule itself, or "anything else" below names nothing.
    assert PREFERENCE_OPERANDS in block, "a preference the engine refuses zeroes the answer"
    # The whole sentence, not its tail. `UNREADABLE` ends in the same six words, so
    # `"scores the whole answer zero" in block` passed with this clause deleted -- the one
    # clause the split of `PREFERENCE_OPERANDS` from `PREFERENCE_DROPPED` exists to get
    # right. A subject-bearing assertion is the only kind that can tell the two apart.
    assert "A preference naming anything else scores the whole answer zero." in block
    # And not `PREFERENCE_DROPPED`: nothing is dropped here, the answer is.
    assert PREFERENCE_DROPPED not in block
    assert "counts as a directive used" not in block


def test_preference_construction_states_the_rule_that_zeroes_its_answer():
    """It sets `min_directives` and so hits the same bloat factor, and never said so."""
    it = pc.make_item(9, 0, "last_link_elitist")
    assert it is not None and it.min_directives
    assert "more than twice" in it.prompt
