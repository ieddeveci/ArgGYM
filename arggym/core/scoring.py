from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import ASPICFramework, Operation
from arggym.core.answers import ScoreResult, UnparseableAnswer
from arggym.core.invariants import split_atoms_and_rules

BLOAT_FACTOR = 2
PARTIAL_CAP = 0.25

#: The rule-name grammar, written once. Every prompt states it in words ("a name
#: starts with a letter and continues with letters, digits or underscores"), the
#: text parser enforces it below, and `check_legality` enforces it again for
#: answers that arrive as values and never pass a parser at all.
_NAME = r"[A-Za-z]\w*"
_LEGAL_NAME = re.compile(rf"^{_NAME}$")

_PREMISE = re.compile(r"^\[(premise|axiom)\s*:\s*(-?[A-Za-z]\w*)\]$")
# The name is required and separated from the kind by whitespace, and the arrow has to
# be the one that kind uses. Written `\s*` with an optional name, the pattern read
# `[stricttest: a => b]` as a strict rule named "test", accepted a rule with no name at
# all under an invented one, and took either arrow for either kind (#19).
_RULE = re.compile(rf"^\[(defeasible|strict)\s+({_NAME})\s*:\s*(.+?)\s*(=>|->)\s*"
                   r"(-?[A-Za-z]\w*)\]$")
ARROW = {"defeasible": "=>", "strict": "->"}
_PREF = re.compile(r"^\[prefer_(rule|premise)\s*:\s*(-?[A-Za-z]\w*)\s*>\s*(-?[A-Za-z]\w*)\]$")


_FENCE_LINE = re.compile(r"^[ \t]*`{3,}\w*[ \t]*$", re.M)


def unmark(text: str) -> str:
    """The answer without markdown code markup: fence lines and backticks.

    A chat model wraps anything that looks like code in a fence or in backticks, and
    the prompts print the answer format in backticks themselves. That markup is
    rendering, like a bullet or a list number, so every parser drops it before reading.
    A line that is only a backtick fence goes, with or without a language tag; then
    every backtick goes. The tag is one word (`\\w*`) by choice, so a line such as
    ```` ```[premise: a] ```` is not a fence line and keeps its content; a tilde
    fence, or a tag with a space in it, is not forgiven.

    No legal token contains a backtick, so markup around a token comes off and
    leaves the token: a stray word in backticks is still a stray word. A backtick
    between two word characters, which is not markdown, glues them instead, so
    `[premise: a`b]` reads as `[premise: ab]`. A space in its place would be
    worse, since it would break `[defeasible k1: a => `b`]`.
    """
    return _FENCE_LINE.sub("", text or "").replace("`", "")


@dataclass
class ParsedAnswer:
    ops: List[Operation] = field(default_factory=list)
    n_lines: int = 0
    n_unparseable: int = 0
    unparseable_examples: List[str] = field(default_factory=list)


def parse_answer(text: str) -> ParsedAnswer:
    """The directives an answer carries.

    The input is the answer, not a completion. Composing the prompt and pulling the
    answer out of whatever the solver returned is the harness's job, so nothing here
    strips answer delimiters -- which is what lets a caller use any convention, or
    none at all (`docs/dataset-contract.md` section 1). Markdown code markup inside
    the answer is another matter: `unmark` drops it, as the parsers drop bullets and
    numbering.
    """
    out = ParsedAnswer()
    body = unmark(text)
    units = re.findall(r"\[[^\]]*\]", body)
    leftover = re.sub(r"\[[^\]]*\]", " ", body)
    stray = [t for t in leftover.split()
             if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    for raw in units + stray:
        line = raw.strip()
        if not line:
            continue
        out.n_lines += 1
        mm = _PREMISE.match(line)
        if mm:
            out.ops.append(Operation(kind=mm.group(1), content=mm.group(2)))
            continue
        mm = _RULE.match(line)
        if mm:
            kind, name = mm.group(1), mm.group(2)
            ants = tuple(a.strip() for a in mm.group(3).split("AND") if a.strip())
            if not ants or mm.group(4) != ARROW[kind]:
                out.n_unparseable += 1
                out.unparseable_examples.append(line[:60])
                continue
            out.ops.append(Operation(kind=kind, name=name,
                                     antecedents=ants, consequent=mm.group(5)))
            continue
        mm = _PREF.match(line)
        if mm:
            out.ops.append(Operation(
                kind="prefer_rule" if mm.group(1) == "rule" else "prefer_premise",
                stronger=mm.group(2), weaker=mm.group(3)))
            continue
        out.n_unparseable += 1
        if len(out.unparseable_examples) < 3:
            out.unparseable_examples.append(line[:60])
    return out


def check_legality(ops: Sequence[Operation], base_ops: Sequence[Operation],
                   allow_strict: bool = False,
                   prefs_only: bool = False) -> Tuple[List[Operation], List[str]]:
    ordinary = {o.content for o in base_ops if o.kind == "premise"}
    axioms = {o.content for o in base_ops if o.kind == "axiom"}
    rule_names = {o.name for o in base_ops if o.kind in ("strict", "defeasible")}
    # #89: a rule named after an atom of the theory is not caught here or anywhere
    # downstream, because `-<name>` is read as the undercut of rule `<name>`
    # (NOTATION.md) whether or not an atom of that name also exists. The rule builds,
    # the answer applies, and the atom's own contrary and preferences now resolve
    # against the rule's on/off switch instead -- a perfect answer can lose this way
    # with `illegal: []`, indistinguishable from reasoning badly. Separating the two
    # namespaces would need new undercut syntax on every prompt to fix a scorer bug, so
    # the collision is rejected instead, the same way a duplicate rule name already is.
    #
    # The answer's own atoms count. Read from `base_ops` alone this check missed a rule
    # named after an atom the answer itself introduced a line earlier -- `[defeasible
    # a1: p => zz]` followed by `[defeasible zz: q => k]` was kept with no reason given,
    # which is the silent zero this check exists to remove. Both lists go in, so the
    # verdict does not depend on which line introduced the atom.
    atoms, _ = split_atoms_and_rules(list(base_ops) + list(ops))
    # NOTATION.md: "rule antecedents must be literals ALREADY present in the theory".
    # Nothing checked it, so a rule over two invented literals was accepted and an answer
    # could route its chain through an intermediate the theory never mentions (#35). A
    # contrary the answer legally introduces counts as present from that line on, which
    # is what makes `[premise: -x]` followed by a rule over -x a legal pair.
    known = set(ordinary) | set(axioms)
    for o in base_ops:
        if o.kind in ("strict", "defeasible"):
            known.update(o.antecedents)
            if o.consequent:
                known.add(o.consequent)
    # The contraries the answer legally introduces are collected first, so a rule may use
    # one whether it is written above or below the premise that introduces it. Checked in
    # a single pass, `[premise: -q]` before a rule over `-q` was legal and the same two
    # lines the other way round were not -- an order nothing else in the pipeline cares
    # about, since `build_framework` sorts operations by kind before applying them.
    for o in ops:
        if o.kind == "premise" and o.content.startswith("-") and o.content[1:] in ordinary \
                and o.content[1:] not in axioms:
            known.add(o.content)
    kept: List[Operation] = []
    reasons: List[str] = []
    for o in ops:
        if o.kind == "strict" and not allow_strict:
            reasons.append(f"illegal_strict_rule:{o.name or '?'}")
            continue
        if prefs_only and o.kind not in ("prefer_rule", "prefer_premise"):
            reasons.append(f"illegal_non_preference:{o.kind}")
            continue
        if o.kind in ("strict", "defeasible"):
            if o.name in rule_names:
                reasons.append(f"duplicate_rule_name:{o.name}")
                continue
            # A value-path answer never meets `_RULE`, so this is the only place the
            # name grammar is enforced for it. Without it a rule named `-ip7` passed
            # both checks below -- `atoms` holds names sign-stripped, so it matched
            # nothing -- and flipped `-ip7` from OVERRULED to JUSTIFIED with an empty
            # `illegal` list.
            if not _LEGAL_NAME.match(o.name or ""):
                reasons.append(f"illegal_rule_name:{o.name}")
                continue
            if o.name in atoms:
                reasons.append(f"illegal_name_collides_with_atom:{o.name}")
                continue
            unknown = [a for a in o.antecedents if a not in known]
            if unknown:
                reasons.append(f"illegal_unknown_antecedent:{','.join(sorted(unknown))}")
                continue
            rule_names.add(o.name)
        if o.kind in ("premise", "axiom"):
            c = o.content
            if o.kind == "axiom":
                reasons.append(f"illegal_new_axiom:{c}")
                continue
            if not c.startswith("-"):
                reasons.append(f"illegal_new_premise:{c}")
                continue
            base = c[1:]
            if base in axioms:
                reasons.append(f"illegal_undermine_axiom:{c}")
                continue
            if base not in ordinary:
                reasons.append(f"illegal_asserted_contrary:{c}")
                continue
        kept.append(o)
    return kept, reasons


def build_framework(base_ops: Sequence[Operation], kept: Sequence[Operation],
                    ordering: str) -> Tuple[ASPICFramework, List[str], List[str]]:
    plain = [o for o in kept if o.kind not in ("prefer_rule", "prefer_premise")]
    prefs = [o for o in kept if o.kind in ("prefer_rule", "prefer_premise")]
    fw = ASPICFramework(ordering=ordering)
    fw.apply_all(list(base_ops) + plain)
    reasons: List[str] = []
    details: List[str] = []
    for o in prefs:
        try:
            fw.apply(o)
        except ValueError as e:
            reasons.append(f"rejected_preference:{o.stronger}>{o.weaker}")
            details.append(str(e))
    return fw, reasons, details


def subgoals_from(goals: Sequence[Dict], base_ops: Sequence[Operation]) -> List[str]:
    """The literals an OVERRULED goal rests on.

    Partial credit for the mixed task uses these: an answer that defeated the
    supports of a claim without overruling the claim itself has done part of the
    work. It is a function of the goals and the theory, so a row records neither
    -- it recomputes here, and a stored copy cannot drift from the definition.
    """
    out: List[str] = []
    for g in goals:
        if g.get("want") != "OVERRULED":
            continue
        for o in base_ops:
            if o.kind in ("defeasible", "strict") and o.consequent == g["claim"]:
                for a in (o.antecedents or ()):
                    if a not in out:
                        out.append(a)
    return out


def _base_status(base_ops: Sequence[Operation], ordering: str,
                 literals: Sequence[str]) -> Dict[str, str]:
    """Each literal's status on the theory before any answer."""
    fw = ASPICFramework(ordering=ordering)
    fw.apply_all(list(base_ops))
    v = ASPICVerifier(fw, operations_applied=len(base_ops))
    return {lit: str(v.status(lit)) for lit in dict.fromkeys(literals)}


def parse(text: str, item: Dict) -> List[Operation]:
    """The directives an answer submits, as values.

    A solver that emits operations directly -- through a JSON schema, a tool call or
    constrained decoding -- skips this and hands the list to `score_value`, which is
    the point of the seam: submitting an answer must not mean imitating our
    serialization (`docs/dataset-contract.md` section 4).

    An empty list is an answer, not a failure: it says the solver submitted nothing,
    and `score_value` reports that. Only text that cannot be read at all raises.
    """
    p = parse_answer(text)
    if p.n_unparseable:
        raise UnparseableAnswer(
            f"unparseable_tokens:{p.n_unparseable}",
            {"n_lines": p.n_lines, "n_unparseable": p.n_unparseable,
             "unparseable_examples": p.unparseable_examples})
    return p.ops


def score_item(answer_text: str, item: Dict) -> ScoreResult:
    """Score one construction answer written as text.

    `score_value(parse(text, item), item)`, with a parse failure turned back into the
    zero result a harness scoring a whole taskset needs.
    """
    try:
        ops = parse(answer_text, item)
    except UnparseableAnswer as e:
        diag: Dict = {"n_lines": 0, "n_unparseable": 0, "illegal": [],
                      "rejected_detail": [], "goals_met": [], "n_used": 0,
                      "minimum": item.get("min_directives")}
        diag.update(e.diagnostics)
        return ScoreResult(score=0.0, success=False, reason=e.reason, diagnostics=diag)
    return score_value(ops, item)


def score_value(ops: Sequence[Operation], item: Dict) -> ScoreResult:
    """Score one construction answer, however it was expressed.

    `success` is the contract's definition for these six tasks -- every goal met and the
    theory left consistent -- and it is returned on every path. An answer that never
    reaches the goal check has not met the goals, so its `success` is False; returning
    nothing there is what made `success_rate` an average over the rows that happened to
    reach a late branch (`docs/dataset-contract.md` section 5).
    """
    ops = list(ops)
    base_ops = item["base_ops"]
    ordering = item["ordering"]
    goals: List[Dict] = item["goals"]
    minimum = item.get("min_directives")

    # Every directive submitted counts against economy, so a repeated one is not free.
    # On the text path this equals the number of readable lines, since a line that could
    # not be read has already raised.
    n_used = len(ops)
    diag: Dict = {"n_lines": n_used, "n_unparseable": 0, "unparseable_examples": [],
                  "illegal": [], "rejected_detail": [], "goals_met": [], "n_used": 0,
                  "minimum": minimum}

    def failed(reason: str, score: float = 0.0) -> ScoreResult:
        return ScoreResult(score=score, success=False, reason=reason, diagnostics=diag)

    if not ops:
        # An answer with nothing in it, rather than an answer that failed to arrive:
        # the harness owns delivery, so there is no such outcome as a missing answer.
        return failed("no_directives")

    kept, illegal = check_legality(ops, base_ops, item.get("allow_strict", False),
                                   item.get("preferences_only", False))
    diag["illegal"] = illegal
    diag["n_used"] = n_used
    if not kept:
        return failed("all_directives_illegal")
    if minimum and n_used > BLOAT_FACTOR * max(minimum, 1):
        diag["bloat_ratio"] = round(n_used / max(minimum, 1), 2)
        return failed(f"bloated:{n_used}_used_vs_{minimum}_minimum")

    try:
        fw, rejected, detail = build_framework(base_ops, kept, ordering)
        illegal.extend(rejected)
        diag["rejected_detail"] = detail
        if len(rejected) == len(kept):
            return failed("all_directives_illegal")
        v = ASPICVerifier(fw, operations_applied=len(base_ops) + len(kept) - len(rejected))
        consistent = v.is_consistent()
    except Exception as e:
        return failed(f"engine_rejected:{type(e).__name__}")

    met = []
    for g in goals:
        got = str(v.status(g["claim"]))
        met.append(got == g["want"])
        diag["goals_met"].append({"claim": g["claim"], "want": g["want"], "got": got})
    success = all(met) and consistent

    diag["achieved_status"] = {g["claim"]: g["got"] for g in diag["goals_met"]}
    # A goal left UNDECIDED where a pole was wanted is unmet, so on success this is 0.
    diag["deadlock_not_defeat"] = 0

    if not success:
        # An answer that reached every goal and broke the theory is not a
        # goal failure, and reporting it as one sends a reader looking at the
        # wrong half of their answer. The prompt states this rule; the reason
        # string has to name it.
        why = "inconsistent_theory" if all(met) else "goal_not_met"
        if not consistent:
            # An inconsistent theory earns no partial credit, so there is no
            # progress to measure and no reason to pay for the base pass below.
            return failed(why)

        # Partial credit pays for movement from where the theory started toward what
        # was asked, so it needs the starting status, which the row does not store
        # (`rows._enc_goals` drops it). One engine pass on the base theory gives it,
        # and only an answer that failed pays for that pass. Read off the final theory
        # alone, an answer that changed nothing collected credit for every goal that
        # started UNDECIDED or started already met (#188).
        start = _base_status(base_ops, ordering,
                             [g["claim"] for g in goals] + list(item.get("subgoals") or []))
        # A deadlock counts only where the answer made it: a goal that started
        # UNDECIDED and is still UNDECIDED is not a conflict the answer created.
        diag["deadlock_not_defeat"] = sum(
            1 for g in diag["goals_met"]
            if g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED")
            and start.get(g["claim"]) != "UNDECIDED")

        # A subgoal counts if it started JUSTIFIED, so the answer had a support to
        # defeat, or if the answer moved it. It is defeated if it ends not JUSTIFIED.
        # One that never stood and was left alone is not the answer's doing; one the
        # answer pushed to JUSTIFIED counts against it.
        sub_counted = sub_hit = 0
        for lit in item.get("subgoals") or []:
            try:
                now = str(v.status(lit))
            except Exception:
                now = None
            if start.get(lit) != "JUSTIFIED" and now == start.get(lit):
                continue
            sub_counted += 1
            sub_hit += now is not None and now != "JUSTIFIED"
        sub_progress = None
        if sub_counted:
            sub_progress = sub_hit / sub_counted
            diag["subgoals_defeated"] = f"{sub_hit}/{sub_counted}"

        per_goal = []
        for g in diag["goals_met"]:
            if start.get(g["claim"]) == g["want"]:
                # Met before the answer. Still met, the answer had no work to do on it,
                # so it leaves the average rather than counting as 1.0. Broken, it
                # stays in at 0.0: moving a goal away from its want earns nothing.
                if g["got"] != g["want"]:
                    per_goal.append(0.0)
                continue
            if g["got"] == g["want"]:
                per_goal.append(1.0)
            elif (g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED")
                  and start.get(g["claim"]) != "UNDECIDED"):
                per_goal.append(0.4)
            elif sub_progress is not None and g["want"] == "OVERRULED":
                per_goal.append(0.3 * sub_progress)
            else:
                per_goal.append(0.0)
        progress = (sum(per_goal) / len(per_goal)) if per_goal else 0.0
        diag["progress"] = round(progress, 4)
        return failed(why, score=round(PARTIAL_CAP * progress, 4))
    if not minimum:
        # Every goal met, so the task is done; the minimum is unknown, so economy cannot
        # be measured. Success without the efficiency half of the score.
        return ScoreResult(score=0.5, success=True, reason="success_but_minimum_unknown",
                           diagnostics=diag)
    efficiency = min(1.0, minimum / max(n_used, 1))
    diag["efficiency"] = round(efficiency, 4)
    return ScoreResult(score=round(0.5 + 0.5 * efficiency, 4), success=True, reason="ok",
                       diagnostics=diag)
