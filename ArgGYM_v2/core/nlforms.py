from __future__ import annotations

from typing import Dict, List

PREMISE: List[str] = [
    "{p} is put forward as a fallible premise",
    "we start from {p}, which is not beyond question",
    "{p} is an uncertain premise of the argument",
    "{p} is accepted provisionally",
    "we may take it that {p}, in the absence of reasons to the contrary",
    "{p} is presumed, pending objection",
    "{p} is granted for now",
    "{p} stands unless something is said against it",
    "it is believed that {p}",
    "{p} is taken to hold",
    "we assume {p}",
    "there is reason to accept {p}",
    "{p} is granted for the sake of argument",
    "{p} is accepted, though it remains open to challenge",
    "{p} is a defeasible starting point",
    "{p} is conceded, subject to challenge",
    "{p} is one of the argument's contestable assumptions",
    "we proceed on the assumption that {p}",
]

AXIOM: List[str] = [
    "{p} is an infallible premise",
    "{p} is taken as an axiom of the argument",
    "{p} belongs to the settled background rather than the fallible assumptions",
    "{p} cannot be argued against",
    "no argument can undermine {p}",
    "{p} is immune to challenge",
    "{p} holds and no counter-argument to it is admissible",
    "no counter-argument to {p} is possible",
    "it is certain that {p}",
    "{p} is settled",
    "{p} holds beyond dispute",
    "{p} is not in question",
]

DEFEASIBLE: List[str] = [
    "{p} creates a presumption in favour of {q}",
    "{p} only raises a presumption that {q}",
    "given {p}, {q} may plausibly be taken to hold",
    "{p} shifts the burden of proof onto anyone denying {q}",
    "presumably, if {p} then {q}",
    "{p} makes {q} plausible, though not certain",
    "from {p} one may defeasibly infer {q}",
    "in the absence of reasons to the contrary, {p} yields {q}",
    "{p} suggests {q}",
    "normally, {p} implies {q}",
    "typically, if {p} holds then so does {q}",
    "{p} supports {q}",
    "{p} ordinarily yields {q}, though the step admits exceptions",
]

STRICT: List[str] = [
    "{p} guarantees {q}",
    "{p} deductively yields {q}",
    "the step from {p} to {q} is a strict one",
    "{p} entails {q}",
    "there is no case in which {p} holds and {q} does not",
    "the step from {p} to {q} admits no exceptions",
    "nothing can block the inference from {p} to {q}",
    "{q} holds in every situation where {p} does",
    "if {p} then necessarily {q}",
    "{p} implies {q} by definition",
    "{q} follows from {p} as a matter of logic",
]

PREFER_RULE: List[str] = [
    "{R1} is preferred to {R2}",
    "{R1} outranks {R2}",
    "when {R1} and {R2} conflict, {R1} prevails",
    "{R1} is the stronger inference compared with {R2}",
    "{R2} yields to {R1} in case of conflict",
    "{R1} takes priority over {R2}",
    "in a clash between {R1} and {R2}, the former wins",
]

PREFER_PREMISE: List[str] = [
    "the premise {p} is preferred to the premise {q}",
    "as assumptions, {p} outranks {q}",
    "between {p} and {q}, {p} is the stronger assumption",
    "{p} takes priority over {q} as a starting point",
    "where {p} and {q} conflict, {p} is to be kept",
]

REBUT: List[str] = [
    "there is an argument that {p} does not hold",
    "{r} is a reason against {p}",
    "{r} counts against {p}",
    "from {r} it follows that not {p}",
]

UNDERCUT: List[str] = [
    "{r} attacks {R1} as an inference, not its conclusion",
    "{r} defeats {R1} itself, leaving its conclusion untouched",
    "{r} is a critical objection to {R1} rather than to what it concludes",
    "{r} blocks {R1}",
    "{R1} does not apply when {r} holds",
    "{r} makes {R1} inapplicable",
    "given {r}, {R1} cannot be used",
    "{R1} is unavailable in the presence of {r}",
    "{r} suspends {R1} without arguing against its conclusion",
]

UNDERMINE: List[str] = [
    "the assumption {p} is challenged by {r}",
    "{r} is an argument against accepting {p}",
    "{r} disputes the premise {p}",
]

NEGATED_LITERAL: List[str] = [
    "not {p}",
    "{p} is false",
    "{p} does not hold",
    "it is not the case that {p}",
    "the denial of {p}",
    "{p} fails",
]

RULE_REFERENCE: List[str] = [
    "the step from {p} to {q}",
    "the inference from {p} to {q}",
    "the rule taking {p} to {q}",
    "the move from {p} to {q}",
    "the reasoning that gets from {p} to {q}",
]

INVENTORY: Dict[str, List[str]] = {
    "premise": PREMISE, "axiom": AXIOM, "defeasible": DEFEASIBLE, "strict": STRICT,
    "prefer_rule": PREFER_RULE, "prefer_premise": PREFER_PREMISE,
    "rebut": REBUT, "undercut": UNDERCUT, "undermine": UNDERMINE,
    "negated_literal": NEGATED_LITERAL, "rule_reference": RULE_REFERENCE,
}


def coverage_report() -> str:
    lines = ["surface forms per construct:"]
    for k, v in INVENTORY.items():
        lines.append(f"  {k:18} {len(v):3}")
    lines.append(f"  {'TOTAL':18} {sum(len(v) for v in INVENTORY.values()):3}")
    return "\n".join(lines)


def shared_vocabulary() -> Dict[str, List[str]]:
    import collections
    where = collections.defaultdict(set)
    for construct, forms in INVENTORY.items():
        for f in forms:
            for w in f.replace("{p}", " ").replace("{q}", " ").replace("{r}", " ") \
                      .replace("{R1}", " ").replace("{R2}", " ").split():
                w = w.strip(",.-").lower()
                if len(w) > 3:
                    where[w].add(construct)
    return {w: sorted(cs) for w, cs in sorted(where.items()) if len(cs) > 1}


OPENERS: List[str] = [
    "Consider the following argument.",
    "The argument runs as follows.",
    "Take the following line of reasoning.",
    "The case is put like this.",
]

LINE_TRANSITIONS: List[str] = [
    "There is also another line of argument for {q}.",
    "A second line of argument reaches {q} as follows.",
    "Separately, {q} can be reached another way.",
    "Independently of the above, there is a further route to {q}.",
    "Another strand of the argument bears on {q}.",
]

NEW_TOPIC: List[str] = [
    "Turning to a separate matter.",
    "A different part of the argument concerns the following.",
    "Set that aside for a moment.",
    "There is a further, unrelated strand.",
]

FORWARD_CONNECTIVES: List[str] = [
    "From this, ", "On that basis, ", "Given this, ", "It follows that ",
    "Accordingly, ", "From that, ",
]

OBJECTION_CONNECTIVES: List[str] = [
    "However, ", "Against this, ", "But ", "There is an objection, though: ",
    "This is not the end of the matter, though: ",
]

RULE_ANAPHORA: List[str] = [
    "that step",
    "this inference",
    "the step just given",
    "that inference",
    "the move just made",
]


# ---------------------------------------------------------------------------
# NEGATION-CARRYING CONSTRUCTS
#
# An audit found `formalization` prose carried almost no negation: zero negated premises, zero rebuts,
# zero negated antecedents. The only negation was the word "not" inside an undercut template, so a model
# never had to WRITE a negated literal at all.
#
# The four forms below all verified against the engine before being added.
# ---------------------------------------------------------------------------

# [axiom: -p] -- an axiom of IMPOSSIBILITY. Verified: it makes p unsatisfiable, and an ordinary premise
# later asserting p is OVERRULED with no preference required. An axiom beats an assertion outright.
NEGATED_AXIOM: List[str] = [
    "it is impossible that {p}",
    "{p} cannot hold under any circumstances",
    "{p} is ruled out as a matter of necessity",
    "there is no possible case in which {p}",
    "{p} is excluded by the settled background",
    "the falsity of {p} is not open to argument",
]

# [premise: -p] -- a FALLIBLE denial. Verified: against an assertion of p this deadlocks unless a
# premise preference decides it, which is the contrast with the axiom above.
NEGATED_PREMISE: List[str] = [
    "{p} is denied",
    "we take it that {p} does not hold",
    "the argument assumes {p} is false",
    "{p} is rejected, though the rejection is itself contestable",
    "there is reason to think {p} fails",
]

# [defeasible r: p => -q] -- a REBUT, an attack on a conclusion rather than on an inference.
# Modgil and Prakken classify this as an attack "on the conclusions of defeasible inferences".
REBUT_RULE: List[str] = [
    "{p} is a reason against {q}",
    "{p} counts against {q}",
    "from {p} one may infer that {q} does not hold",
    "{p} tells against {q}",
    "{p} raises a presumption that {q} is false",
]

# [strict r: p -> -q] -- an exclusion. Strict, so it cannot be undercut and wins outright.
STRICT_EXCLUSION: List[str] = [
    "{p} rules out {q}",
    "{p} excludes {q} entirely",
    "{q} cannot hold wherever {p} does",
    "{p} is incompatible with {q}",
]

INVENTORY["negated_axiom"] = NEGATED_AXIOM
INVENTORY["negated_premise"] = NEGATED_PREMISE
INVENTORY["rebut_rule"] = REBUT_RULE
INVENTORY["strict_exclusion"] = STRICT_EXCLUSION


# [strict r: -p -> q] -- a rule firing FROM a negated literal, strictly.
#
# Verified, and the contrast with the defeasible version is the point: with the same negated antecedent,
# an undercut aimed at the step defeats the conclusion when the step is defeasible and does NOTHING when
# it is strict. Combined with an impossibility axiom the result is unassailable -- not undermineable at
# the root, not undercuttable at the step, and not rebuttable at the conclusion, since ASPIC+ forbids
# rebutting a strictly derived conclusion.
STRICT_FROM_NEGATION: List[str] = [
    "if {p} does not hold then necessarily {q}",
    "the absence of {p} guarantees {q}",
    "wherever {p} fails, {q} follows without exception",
    "the failure of {p} entails {q}",
    "there is no case in which {p} is absent and {q} does not hold",
]

INVENTORY["strict_from_negation"] = STRICT_FROM_NEGATION


# ---------------------------------------------------------------------------
# JUNCTIONS -- rules with two antecedents.
#
# Modgil and Prakken write defeasible rules with a set of antecedents, and the conjunctive case is the
# normal one in argument schemes: a scheme's premises jointly license its conclusion, and knocking out
# any one of them defeats the argument. Every rule in this suite was single-antecedent until this was
# added, so nothing exercised the case where a conclusion has two independent supports and either can
# be cut.
# ---------------------------------------------------------------------------

JUNCTION_DEFEASIBLE: List[str] = [
    "taken together, {p} and {q} support {r}",
    "{p} and {q} jointly give reason to accept {r}",
    "given both {p} and {q}, {r} may be presumed",
    "{p} in combination with {q} creates a presumption in favour of {r}",
    "where {p} and {q} both hold, {r} normally follows",
    "{r} rests on {p} and {q} together, and needs both",
]

JUNCTION_STRICT: List[str] = [
    "{p} and {q} together entail {r}",
    "there is no case in which {p} and {q} hold and {r} does not",
    "{p} and {q} jointly guarantee {r}",
    "taken together, {p} and {q} settle {r} conclusively",
]

INVENTORY["junction_defeasible"] = JUNCTION_DEFEASIBLE
INVENTORY["junction_strict"] = JUNCTION_STRICT
