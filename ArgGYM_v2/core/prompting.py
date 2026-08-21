from __future__ import annotations

from typing import Dict, List, Optional, Sequence

SEMANTICS = "grounded semantics"
ORDERING_NAME = {
    "last_link_elitist": "the last-link strength ordering",
    "weakest_link_elitist": "the weakest-link strength ordering",
}

_PERMITTED = (
    "Permitted additions, written exactly in these forms:\n"
    "   [defeasible <name>: <antecedent> => <consequent>]\n"
    "   [prefer_rule: <rule> > <rule>]\n"
    "   [prefer_premise: <literal> > <literal>]\n"
    "   [premise: -<literal>]   permitted only as the negation of an ordinary premise above\n"
    "Rule antecedents must be literals already present in the theory. "
    "A rule name in a consequent, written -<name>, switches that rule off."
)
_FORMAT = ("Answer format: one directive per line, between [answer] and [/answer].")

# MINIMAL PROMPTS ARE THE DEFAULT.
#
# A benchmark item should carry the QUESTION and nothing else. Explanations of the notation -- what a
# negated rule name means, which additions are legal, how the DSL is written -- belong in a separate
# context document that a researcher supplies or withholds deliberately, because whether the model
# knows the notation is a different variable from whether it can reason.
#
# This is not hypothetical. Measured on one perturbation item: adding a single sentence explaining that
# a negated rule name disables that rule moved RECALL from 0.480 to 0.880, while precision barely
# moved (0.857 to 0.815). Almost the entire earlier failure was notation, not reasoning -- and without
# the split it would have been read as a reasoning result.
#
# What stays in the prompt: the theory, the semantics, the ordering, the question, and the bare answer
# format needed to parse a response at all. What moves out: everything explanatory.
INCLUDE_NOTATION = False

_STATUS_WORD = {"JUSTIFIED": "justified", "OVERRULED": "overruled", "UNDECIDED": "undecided"}


def _goal_line(claim: str, current: str, want: str) -> str:
    cur = _STATUS_WORD.get(current, current.lower())
    tgt = _STATUS_WORD.get(want, want.lower())
    return (f"The claim {claim} is currently {cur}.\n"
            f"What is the minimal set of directives that makes {claim} {tgt}?")


def render(theory_text: str, ordering: str, goals: Sequence[Dict],
           include_notation: Optional[bool] = None) -> str:
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
        parts += ["", _PERMITTED]
    parts += ["", _FORMAT]
    return "\n".join(parts)
