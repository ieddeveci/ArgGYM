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

_STATUS_WORD = {"JUSTIFIED": "justified", "OVERRULED": "overruled", "UNDECIDED": "undecided"}


def _goal_line(claim: str, current: str, want: str) -> str:
    cur = _STATUS_WORD.get(current, current.lower())
    tgt = _STATUS_WORD.get(want, want.lower())
    return (f"The claim {claim} is currently {cur}.\n"
            f"What is the minimal set of directives that makes {claim} {tgt}?")


def render(theory_text: str, ordering: str, goals: Sequence[Dict]) -> str:
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
    parts += ["", _PERMITTED, "", _FORMAT]
    return "\n".join(parts)
