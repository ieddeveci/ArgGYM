
from aspic_gym import GYM_ORDERING, _entry, _norm_syntax

_ONE = ("In the answer, write exactly one directive, in the DSL syntax above, and nothing "
        "else. Replace the placeholders (X, A, B, C, dK) with the actual statements and rule "
        "labels named in this question.")
_STATUSES = ["justified", "overruled", "undecided", "unsatisfiable"]




def _pick(rng, options):
    return options[rng.randrange(len(options))]



def _fam_contract(rng, a, b, c, k):
    kind = rng.randrange(4)
    if kind == 0:                               
        w = _pick(rng, _STATUSES)
        instr = _pick(rng, [
            f'Answer with the single word "{w}".',
            f'The verdict on the claim is {w}; report just that one word.',
            f'Write the status word "{w}" as your final answer.',
        ])
        return instr, w, f"In the answer, write only the word {w}."
    if kind == 1:                                 
        instr = _pick(rng, [
            'Answer with the single word "none".',
            "No claim qualifies; report the word none.",
        ])
        return instr, "none", "In the answer, write only the word none."
    if kind == 2:                                 
        n = rng.randint(1, 5)
        w = _pick(rng, _STATUSES)
        instr = _pick(rng, [
            f"Report that claim {n} is {w}, as a `number: status` line.",
            f"Claim {n} turned out to be {w}; write this in the `number: status` line format.",
        ])
        return instr, f"{n}: {w}", "In the answer, write exactly one line of the form `number: status`."
    n1 = rng.randint(1, 3); n2 = rng.randint(n1 + 1, 6)        
    instr = _pick(rng, [
        f"Claims {n1} and {n2} are established; list their numbers separated by a comma.",
        f"Exactly claims {n1} and {n2} qualify. Answer with the two numbers, comma-separated, "
        f"in ascending order.",
    ])
    return instr, f"{n1}, {n2}", "In the answer, write only the numbers, separated by a comma."


def _fam_facts(rng, a, b, c, k):
    kind = rng.randrange(3)
    if kind == 0:                               
        instr = _pick(rng, [
            f'State the fact "{a}" as a directive.',
            f'Record "{a}" as an ordinary, challengeable fact.',
            f'Add the premise "{a}" to the theory.',
        ])
        return instr, f"[premise: {a}]", _ONE
    if kind == 1:                                  
        instr = _pick(rng, [
            f'State the certain (axiomatic) fact "{a}" as a directive.',
            f'Record "{a}" as a fact that is beyond challenge.',
            f'"{a}" is taken as absolutely certain; add it to the theory accordingly.',
        ])
        return instr, f"[axiom: {a}]", _ONE
    instr = _pick(rng, [                            
        f'State, as an ordinary fact, that "{a}" does NOT hold.',
        f'Add the negation of "{a}" to the theory as a premise.',
    ])
    return instr, f"[premise: -{a}]", _ONE


def _fam_rules(rng, a, b, c, k):
    kind = rng.randrange(6)
    if kind == 0:                                
        instr = _pick(rng, [
            f'Write the rule "if {a} then {b}" (a rule open to exceptions) as a directive.',
            f'"{a}" normally brings about "{b}" (exceptions are possible). Write this rule.',
        ])
        return instr, f"[defeasible: {a} => {b}]", _ONE
    if kind == 1:                                  
        instr = _pick(rng, [
            f'Write the rule "if {a} and {b} then {c}" (open to exceptions) as a directive.',
            f'When both "{a}" and "{b}" hold, "{c}" follows -- normally, though not without exception. Write this rule.',
        ])
        return instr, f"[defeasible: {a} AND {b} => {c}]", _ONE
    if kind == 2:                                   
        instr = _pick(rng, [
            f'Write the rule "if {a} then {b}, necessarily" as a directive.',
            f'"{a}" guarantees "{b}" without exception. Write this rule.',
        ])
        return instr, f"[strict: {a} -> {b}]", _ONE
    if kind == 3:                               
        instr = _pick(rng, [
            f'Write the rule "if {a} and {b} then {c}, necessarily" as a directive.',
            f'Together, "{a}" and "{b}" necessarily entail "{c}". Write this rule.',
        ])
        return instr, f"[strict: {a} AND {b} -> {c}]", _ONE
    if kind == 4:                               
        instr = _pick(rng, [
            f'Write the rule "if {a} then NOT {b}" (a rule open to exceptions) as a directive.',
            f'"{a}" normally rules "{b}" out. Write this as a defeasible rule concluding the '
            f'negation of "{b}".',
        ])
        return instr, f"[defeasible: {a} => -{b}]", _ONE
    direction = rng.random() < 0.5               
    if direction:
        instr = _pick(rng, [
            f"The rule [defeasible: {a} => {b}] should hold necessarily, without exception. "
            f"Rewrite it as the correct directive.",
            f"Upgrade [defeasible: {a} => {b}] into a rule that admits no exception.",
        ])
        return instr, f"[strict: {a} -> {b}]", _ONE
    instr = _pick(rng, [
        f"The rule [strict: {a} -> {b}] is too strong; it should be defeasible -- open to exceptions. "
        f"Rewrite it as the correct directive.",
        f"Downgrade [strict: {a} -> {b}] into a rule that can be defeated.",
    ])
    return instr, f"[defeasible: {a} => {b}]", _ONE


def _fam_attacks(rng, a, b, c, k):
    kind = rng.randrange(4)
    if kind == 0:                                   
        instr = _pick(rng, [
            f'An argument concludes "{b}". Rebut it: assert the contrary of that conclusion '
            f'as a premise.',
            f'To attack the conclusion "{b}" head-on, state its negation as a fact.',
        ])
        return instr, f"[premise: -{b}]", _ONE
    if kind == 1:                                 
        instr = _pick(rng, [
            f'An argument stands on the ordinary premise "{a}". Undermine it: assert the '
            f'negation of that premise.',
            f'Attack the argument at its foundation by denying its premise "{a}".',
        ])
        return instr, f"[premise: -{a}]", _ONE
    if kind == 2:                                 
        instr = _pick(rng, [
            f"Write a directive that undercuts rule d{k} (switch it off by asserting the "
            f"negation of its label).",
            f"Rule d{k} should not apply here. Disable it with a single premise directive.",
        ])
        return instr, f"[premise: -d{k}]", _ONE
    instr = _pick(rng, [                           
        f'Write a defeasible rule that undercuts rule d{k} whenever "{a}" holds.',
        f'When "{a}" holds, rule d{k} should be switched off. Write this as a defeasible rule '
        f"concluding the negation of the rule's label.",
    ])
    return instr, f"[defeasible: {a} => -d{k}]", _ONE


def _fam_preferences(rng, a, b, c, k):
    kind = rng.randrange(3)
    if kind == 0:                              
        instr = _pick(rng, [
            f"State that rule d{k} is stronger than rule d{k + 1}.",
            f"Rules d{k} and d{k + 1} conflict; settle the tie in favour of d{k}.",
        ])
        return instr, f"[prefer_rule: d{k} > d{k + 1}]", _ONE
    if kind == 1:                                   
        instr = _pick(rng, [
            f"State that rule d{k} is WEAKER than rule d{k + 1}.",
            f"Rule d{k} must lose to rule d{k + 1}; write the preference that says so.",
        ])
        return instr, f"[prefer_rule: d{k + 1} > d{k}]", _ONE
    instr = _pick(rng, [                         
        f'State that the premise "{a}" is stronger than the premise "{b}".',
        f'The premises "{a}" and "{b}" clash; declare "{a}" the more reliable one.',
    ])
    return instr, f"[prefer_premise: {a} > {b}]", _ONE


def _fam_recognition(rng, a, b, c, k):
    kind = rng.randrange(3)
    if kind == 0:                                   
        shown, word = _pick(rng, [
            (f"[premise: {a}]", "premise"),
            (f"[axiom: {a}]", "axiom"),
            (f"[defeasible: {a} => {b}]", "defeasible"),
            (f"[strict: {a} -> {b}]", "strict"),
        ])
        instr = (f"What kind of directive is {shown}? Answer with one word: "
                 "premise, axiom, defeasible, or strict.")
        return instr, word, "In the answer, write only that one word."
    if kind == 1:                                
        instr = (f"Which rule does the directive [premise: -d{k}] switch off? "
                 f"Answer with the rule label alone.")
        return instr, f"d{k}", "In the answer, write only the rule label."
    instr = (f'What is the negation of the statement "{a}"? '           
             f"Answer with the negated statement alone.")
    return instr, f"-{a}", "In the answer, write only the negated statement."


def _fam_sequence(rng, a, b, c, k):
    kind = rng.randrange(2)
    if kind == 0:
        instr = _pick(rng, [
            f'State the fact "{a}", then the default rule that derives "{b}" from it - two '
            f"directives, in that order.",
            f'First add "{a}" as a premise; then add the defeasible rule "if {a} then {b}". '
            f"Write both directives, one per line, in that order.",
        ])
        gold = f"[premise: {a}]\n[defeasible: {a} => {b}]"
    else:
        instr = _pick(rng, [
            f'Assert "{a}" as certain, then state that "{a}" necessarily entails "{b}" - two '
            f"directives, in that order.",
            f'First add "{a}" as an axiom; then add the strict rule "if {a} then {b}". '
            f"Write both directives, one per line, in that order.",
        ])
        gold = f"[axiom: {a}]\n[strict: {a} -> {b}]"
    return instr, gold, ("In the answer, write exactly these two directives, one per line, "
                         "in the stated order, and nothing else.")


_FAMILIES = [
    ("contract", _fam_contract),
    ("facts", _fam_facts),
    ("rules", _fam_rules),
    ("attacks", _fam_attacks),
    ("preferences", _fam_preferences),
    ("recognition", _fam_recognition),
    ("sequence", _fam_sequence),
]


def _syntax_items(rng):
    A = "abcdefghijklmnopqrstuvwxyz"
    pool = list(A) + [ch + str(d) for ch in A for d in range(10)]
    a, b, c = rng.sample(pool, 3)
    k = rng.randint(1, 40)
    fam, builder = _FAMILIES[rng.randrange(len(_FAMILIES))]
    instr, gold, spec = builder(rng, a, b, c, k)
    return fam, instr, gold, spec


_DEMOS = {
    "directive": (
        "Example (a different, unrelated instruction, showing the required format):\n"
        "Instruction: Write the rule \"if p then q\" (open to exceptions) as a directive.\n"
        "Solution:\n[reasoning]\n"
        "A default rule is a defeasible rule; its arrow is =>.\n"
        "[/reasoning]\n[answer]\n[defeasible: p => q]\n[/answer]\n\n"
    ),
    "token": (
        "Example (a different, unrelated instruction, showing the required format):\n"
        "Instruction: Answer with the single word \"overruled\".\n"
        "Solution:\n[reasoning]\n"
        "The task asks for exactly one status word.\n"
        "[/reasoning]\n[answer]\noverruled\n[/answer]\n\n"
    ),
}


_SYNTAX_INTRO = (
    "You are translating between plain statements and the directive syntax (DSL) of "
    "defeasible argumentation. Your job is to write, or identify, a single directive in "
    "the DSL.\n\n"
    "The directives:\n"
    "  [premise: X] -- states that X is an ordinary fact: something taken as true for now, "
    "but which another argument is allowed to challenge or defeat.\n"
    "  [axiom: X] -- states that X is a certain fact: something taken as true absolutely, "
    "which no argument may challenge.\n"
    "  [defeasible dK: A AND B => C] -- a rule that holds BY DEFAULT: when A and B are true, "
    "C normally follows, but the rule can be overridden by a stronger or conflicting "
    "argument. The => arrow marks a defeasible (defeatable) rule. dK is its label (d1, d2, ...).\n"
    "  [strict sK: A AND B -> C] -- a rule that holds NECESSARILY: when A and B are true, C "
    "must follow, with no exception. The -> arrow marks a strict rule. sK is its label.\n"
    "  [prefer_rule: d1 > d2] -- states that rule d1 is stronger than rule d2, so d1 wins if "
    "they conflict.\n"
    "  [prefer_premise: x > y] -- states that premise x is stronger than premise y.\n\n"
    "Notation: the negation (opposite) of X is written -X. To undercut a rule dK (switch it "
    "off) you write -dK. Join multiple conditions in a rule with AND."
)


def _syntax_format_block(spec: str) -> str:
    return ("Task: " + spec + "\n\n"
            "Think step by step about how to express the answer in the DSL directive syntax, "
            "or, if the task only asks for a single word, about which word is correct. When "
            "you have finished thinking, write the [/reasoning] tag. Then give your final "
            "answer between [answer] and [/answer] tags, and stop.\n\n"
            "Solution:\n[reasoning]\n")


def _syntax_exemplar(family):
    return _DEMOS["directive"] if family in ("contract", "recognition") else _DEMOS["token"]


def frame_syntax(rng, with_content=False, level=1):
    fam, instr, gold, spec = _syntax_items(rng)
    prompt = (_syntax_exemplar(fam) + _SYNTAX_INTRO + "\n\n" + instr + "\n\n"
              + _syntax_format_block(spec + " Put it inside the [answer] tags."))
    reference = "[answer]\n" + gold + "\n[/answer]"
    return _entry("syntax", prompt, reference, [], GYM_ORDERING,
                  target_norm=_norm_syntax(gold), family=fam, mode="symbolic")
