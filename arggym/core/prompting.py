"""What the question says a legal answer must contain.

The notation contract is written here and nowhere else in generation. Four
blocks cover seven task variants, and they differ because the scorer differs:
what an item permits is not a single axis, so `formalization` and
`preference_construction` get their own block rather than a parameter on
`permitted_block`. Each block's docstring names the scoring policy that makes
its clauses the right ones.

`NOTATION.md` states the same rules for a human reader, in its own register.
`tests/test_the_notation_contract_is_stated_where_it_is_enforced.py` holds the
two together.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

SEMANTICS = "grounded semantics"
ORDERING_NAME = {
    "last_link_elitist": "the last-link elitist strength ordering",
    "last_link_democratic": "the last-link democratic strength ordering",
    "weakest_link_elitist": "the weakest-link elitist strength ordering",
    "weakest_link_democratic": "the weakest-link democratic strength ordering",
}

_STRICT_FORM = "   [strict <name>: <antecedent> -> <consequent>]\n"
# The two preference forms, shared because `preference_construction` permits exactly
# these and nothing else. Written twice, they were byte-identical copies.
_PREF_FORMS = ("   [prefer_rule: <rule> > <rule>]\n"
               "   [prefer_premise: <literal> > <literal>]\n")
# The grammar `scoring._NAME` compiles, in words. Two blocks state it, and #89 is what
# happens when a naming clause has to be edited in more than one wording.
_NAME_GRAMMAR = ("name starts with a letter and continues with letters, digits or "
                 "underscores")
# Two rules that `score_item` applies to every task routed through it, including the
# one that writes its own prompt. Naming them here is what stops a second wording
# drifting into existence: `preference_construction` stated neither, and was scored by
# both.
MINIMALITY = ("The answer must be minimal: one using more than twice the fewest "
              "directives that work scores zero.")
UNREADABLE = "A directive that cannot be read at all scores the whole answer zero"

_TAIL = ("Every rule needs a name, written after the kind and separated from it by a space. "
         "A " + _NAME_GRAMMAR + ", "
         "and no rule already in the theory, no earlier line of the answer, and no "
         "atom of the theory or of your answer, has used it. An atom is a name "
         "without its leading -, so -ko1 uses the atom ko1. "
         "The arrow is => for a defeasible rule and -> for a strict one.\n"
         "Several antecedents are joined with AND. Rule antecedents must be literals "
         "already present in the theory. A rule name in a consequent, written -<name>, "
         "switches that rule off.\n"
         + MINIMALITY + " " + UNREADABLE + ".")

# Two contraries are both JUSTIFIED under grounded semantics only if both are firm, and
# firmness comes from an axiom through strict rules. No answer may add an axiom, so an
# answer reaches this branch only where it may add a strict rule. On the other four
# construction tasks the sentence described something their answers cannot do.
INCONSISTENT = ("An answer that reaches every goal while leaving the theory inconsistent "
                "also scores zero.")


def permitted_block(allow_strict: bool = False) -> str:
    """The permitted directive forms, as the scorer actually enforces them.

    `check_legality` rejects a strict rule unless the item allows one, and rejects any
    new axiom or bare premise (`core/scoring.py`). Until now no prompt said so:
    the two `counter_argument` variants rendered byte-identically while one accepted a
    one-directive strict answer and the other scored it zero (#37), and this block
    itself shipped on no item because the flag that gates it was off (#38, #18).

    Its content is the same on every item of a variant, so it leaks nothing about the
    theory it is attached to.
    """
    return ("Permitted additions, written exactly in these forms:\n"
            "   [defeasible <name>: <antecedent> => <consequent>]\n"
            + (_STRICT_FORM if allow_strict else "")
            + _PREF_FORMS
            + "   [premise: -<literal>]   permitted only as the negation of an ordinary "
              "premise written above without a leading -\n"
            + ("" if allow_strict else "Strict rules may not be added.\n")
            + "New axioms may not be added.\n"
            + _TAIL
            + (" " + INCONSISTENT if allow_strict else ""))


def preference_block() -> str:
    """The permitted forms for `preference_construction`, and the rules that zero it.

    Narrower than `permitted_block` because the registry sets `preferences_only`
    (`core/registry.py`), so `check_legality` rejects every directive that is not a
    preference and keeps none of them. A strict rule is rejected a line earlier as
    `illegal_strict_rule` (`core/scoring.py`) and a defeasible one as
    `illegal_non_preference`, but either way the directive is dropped before any naming,
    antecedent or premise check runs. Three clauses drop out as a result:

    - No rule survives, so the naming grammar applies to nothing.
    - No premise or axiom survives, so the premise and axiom clauses apply to nothing.
    - Preferences alone cannot make a consistent theory inconsistent, so the
      consistency sentence describes an outcome this task's answers cannot reach.

    What stays is what `score_item` still applies: the bloat factor, which this task
    hits because it sets `min_directives`, and the unreadable-line rule.
    """
    return ("Permitted additions: preference directives only, written exactly in "
            "these forms:\n"
            + _PREF_FORMS
            + "No new rules or premises may be added.\n"
            + MINIMALITY + "\n"
            + UNREADABLE + ", so write only directives.")


def formalization_notation() -> str:
    """The whole DSL, for the one task whose answer is a theory rather than an addition.

    `formalization` is `GRADED` (`core/registry.py`) and never calls `check_legality`,
    so none of the legality clauses apply. It is asked for a theory rather than for
    additions to one, which is why every kind is permitted here and axioms and strict
    rules are listed alongside the rest.

    There is no theory whose names the answer could collide with, so the grammar is
    stated and the collision clause is not; and the task sets no `min_directives`, so
    the minimality sentence would be a promise the row cannot keep -- `core/freeze.py`
    refuses a taskset that states it without one.
    """
    return ("Notation: [premise: x], [axiom: x], [defeasible name: a => b], "
            "[strict name: a -> b], "
            "[prefer_rule: r1 > r2], [prefer_premise: x > y]. Negation is written -x. "
            "Rule names are yours to choose: a " + _NAME_GRAMMAR + ", "
            "and is separated from the kind by a space.")


_FORMAT = ("Answer format: one directive per line.")


def answer_format(clause: str) -> str:
    """The answer-format block: what the answer must say, and nothing about where.

    The clause is the task's own and describes content -- one directive per line,
    one line per claim. Where the answer goes is delivery, and delivery belongs to
    the harness: it composes the prompt from this question, calls whatever solver it
    likes, and extracts the answer before handing it back. So the question says
    nothing about a fence, on any task and for any caller
    (`docs/dataset-contract.md` section 1).

    One line of work, kept as a function because this is where that rule is written
    for the nine task modules that call it.
    """
    return clause.rstrip()


_STATUS_WORD = {"JUSTIFIED": "justified", "OVERRULED": "overruled", "UNDECIDED": "undecided"}


def _goal_line(claim: str, current: str, want: str) -> str:
    cur = _STATUS_WORD.get(current, current.lower())
    tgt = _STATUS_WORD.get(want, want.lower())
    return (f"The claim {claim} is currently {cur}.\n"
            f"What is the minimal set of directives that makes {claim} {tgt}?")


def render(theory_text: str, ordering: str, goals: Sequence[Dict]) -> str:
    """The question for the three `attack_defense` modes.

    No strict flag. The registry gives those three no `allow_strict`, so `check_legality`
    rejects a strict rule on every one of them; a flag here could only ever offer a form
    the scorer zeroes, which is #37 with the sign flipped and nothing in `core/freeze.py`
    to catch it. `counter_argument` passes its own flag to `permitted_block` directly.
    """
    head = (f"The following is a defeasible argumentation theory, evaluated under {SEMANTICS} "
            f"with {ORDERING_NAME.get(ordering, ordering)}.")
    parts: List[str] = [head, "", theory_text, ""]
    if len(goals) == 1:
        g = goals[0]
        parts.append(_goal_line(g["claim"], g["current"], g["want"]))
    else:
        for g in goals:
            cur = _STATUS_WORD.get(g["current"], g["current"].lower())
            parts.append(f"The claim {g['claim']} is currently {cur}.")
        wants = ", ".join(f"{g['claim']} {_STATUS_WORD.get(g['want'], g['want'].lower())}"
                          for g in goals)
        parts.append("What is the minimal set of directives that simultaneously makes "
                     f"{wants}?")
    parts += ["", permitted_block(), "", answer_format(_FORMAT)]
    return "\n".join(parts)
