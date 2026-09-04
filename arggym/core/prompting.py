from __future__ import annotations

from typing import Dict, List, Optional, Sequence

SEMANTICS = "grounded semantics"
ORDERING_NAME = {
    "last_link_elitist": "the last-link elitist strength ordering",
    "last_link_democratic": "the last-link democratic strength ordering",
    "weakest_link_elitist": "the weakest-link elitist strength ordering",
    "weakest_link_democratic": "the weakest-link democratic strength ordering",
}

_STRICT_FORM = "   [strict <name>: <antecedent> -> <consequent>]\n"
_TAIL = ("Every rule needs a name, written after the kind and separated from it by a space. "
         "A name starts with a letter and continues with letters, digits or underscores, "
         "and no rule already in the theory, and no earlier line of the answer, has used it. "
         "The arrow is => for a defeasible rule and -> for a strict one.\n"
         "Several antecedents are joined with AND. Rule antecedents must be literals "
         "already present in the theory. A rule name in a consequent, written -<name>, "
         "switches that rule off.\n"
         "The answer must be minimal: one using more than twice the fewest directives that "
         "work scores zero. A directive that cannot be read at all scores the whole answer "
         "zero, and so does an answer that reaches every goal while leaving the theory "
         "inconsistent.")


def permitted_block(allow_strict: bool = False) -> str:
    """The permitted directive forms, as the scorer actually enforces them.

    `check_legality` rejects a strict rule unless the item allows one, and rejects any
    new axiom or bare premise (`core/scoring.py:82-104`). Until now no prompt said so:
    the two `counter_argument` variants rendered byte-identically while one accepted a
    one-directive strict answer and the other scored it zero (#37), and this block
    itself shipped on no item because the flag that gates it was off (#38, #18).

    Its content is the same on every item of a variant, so it leaks nothing about the
    theory it is attached to.
    """
    return ("Permitted additions, written exactly in these forms:\n"
            "   [defeasible <name>: <antecedent> => <consequent>]\n"
            + (_STRICT_FORM if allow_strict else "")
            + "   [prefer_rule: <rule> > <rule>]\n"
              "   [prefer_premise: <literal> > <literal>]\n"
              "   [premise: -<literal>]   permitted only as the negation of an ordinary "
              "premise written above without a leading -\n"
            + ("" if allow_strict else "Strict rules may not be added.\n")
            + "New axioms may not be added.\n"
            + _TAIL)


_FORMAT = ("Answer format: one directive per line.")

INCLUDE_NOTATION = True

_STATUS_WORD = {"JUSTIFIED": "justified", "OVERRULED": "overruled", "UNDECIDED": "undecided"}


def _goal_line(claim: str, current: str, want: str) -> str:
    cur = _STATUS_WORD.get(current, current.lower())
    tgt = _STATUS_WORD.get(want, want.lower())
    return (f"The claim {claim} is currently {cur}.\n"
            f"What is the minimal set of directives that makes {claim} {tgt}?")


def render(theory_text: str, ordering: str, goals: Sequence[Dict],
           include_notation: Optional[bool] = None, allow_strict: bool = False) -> str:
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
    if INCLUDE_NOTATION if include_notation is None else include_notation:
        parts += ["", permitted_block(allow_strict)]
    parts += ["", _FORMAT]
    return "\n".join(parts)
