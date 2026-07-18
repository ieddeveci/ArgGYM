_INTRO = (
    "You are reasoning about a defeasible argumentation theory under grounded semantics.\n"
    "Directive syntax:\n"
    "  [premise: X] an ordinary fact; it can be challenged.\n"
    "  [axiom: X] a fact taken as certain; it cannot be challenged.\n"
    "  [defeasible dK: A AND B => C] if A and B hold, C holds by default (the rule can be defeated).\n"
    "  [strict sK: A AND B -> C] if A and B hold, C holds necessarily.\n"
    "  [prefer_rule: d1 > d2] rule d1 is stronger than rule d2.\n"
    "  [prefer_premise: p > q] premise p is stronger than premise q.\n"
    "Attacks. The negation of X is written -X. An argument for X and an argument for -X REBUT each "
    "other; asserting -p against an ordinary premise p UNDERMINES it; deriving -dK UNDERCUTS rule dK "
    "(switches it off). A conclusion that uses no defeasible rule is FIRM and defeats any conflicting "
    "defeasible conclusion.\n"
    "Preferences and defeat. When two conflicting defeasible arguments meet, a preference lets the "
    "stronger prevail and OVERRULE the weaker; with no applicable preference the clash is unresolved "
    "and both stay undecided. A rule preference bites on a rule-level conflict; a premise preference "
    "bites when the conflict is rooted in the premises (an undermining attack, or whenever an "
    "argument's premises are the deciding link).\n"
    "Orderings. How an argument's strength is computed depends on the ordering in force, which each "
    "problem states. Under LAST-LINK ordering, strength is decided by the argument's final defeasible "
    "rule, so a preference on that last rule suffices to win a conflict. Under WEAKEST-LINK ordering, "
    "an argument is only as strong as its weakest link across the whole chain, so a preference wins "
    "only if it outranks the opponent throughout (rules and premises). The two orderings can give "
    "different statuses to the same claim.\n"
    "Statuses: justified (an argument for it survives every attack), overruled (defeated by a "
    "stronger surviving argument), undecided (caught in an unresolved conflict), unsatisfiable "
    "(no argument for it exists)."
)

_SYMBOLIC_NOTATION = (
    "The theory below is written in that directive syntax; rules are labelled dK / sK so that "
    "preferences and undercuts can name them."
)

_CONTENT_NOTATION = (
    "The theory below is written in plain language: given facts first, then rules "
    "(labelled 'Rule k' so preferences and undercuts can name them), then preferences "
    "(between two rules, or between two statements). "
    "The negation of a statement is its explicitly stated opposite."
)


def _intro(level: int) -> str:
    return _INTRO


def _sym_notation(level: int) -> str:
    return _SYMBOLIC_NOTATION


def _ordering_decl(ordering: str) -> str:
    if "weakest_link" in ordering:
        return ("ORDERING: this problem uses WEAKEST-LINK ordering. An argument is only as "
                "strong as its weakest defeasible rule, so a preference makes an argument win a "
                "conflict only if it outranks the opponent across its whole chain (both rules and "
                "premises).\n\n")
    return ("ORDERING: this problem uses LAST-LINK ordering. An argument's strength is decided "
            "by its final (last) defeasible rule, so a preference on that final rule is enough "
            "to make the argument prevail in a conflict.\n\n")


def _format_block(answer_spec: str) -> str:
    return ("Task instructions: " + answer_spec + "\n\n"
            "Answer format: reason step by step first, close the reasoning with [/reasoning], "
            "then give only the final answer between [answer] and [/answer], then stop.\n\n"
            "Solution:\n[reasoning]\n")

_EXEMPLARS = {
    "status_query": (
        "Example (a different, unrelated problem, showing the required format):\n"
        "Problem: [premise: p] [defeasible d1: p => q]. What is the status of claim 1. q?\n"
        "Solution:\n"
        "[reasoning]\n"
        "p is a premise, so there is an argument for p. Rule d1 fires on p and concludes q. "
        "Nothing in the theory attacks p, d1, or q, so the argument for q survives every attack.\n"
        "[/reasoning]\n"
        "[answer]\n"
        "1: justified\n"
        "[/answer]\n\n"
    ),
}


def exemplar(kind: str, level: int) -> str:
    return _EXEMPLARS.get(kind, "") if level == 1 else ""


def assemble(intro: str, task_body: str, format_block: str) -> str:
    return intro + "\n\n" + task_body + "\n\n" + format_block
