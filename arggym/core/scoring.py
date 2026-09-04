from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import ASPICFramework, Operation
from arggym.aspic.api import ASPICVerifier
from arggym.core.answers import ScoreResult, extract_answer

BLOAT_FACTOR = 2
PARTIAL_CAP = 0.25

_PREMISE = re.compile(r"^\[(premise|axiom)\s*:\s*(-?[A-Za-z]\w*)\]$")
# The name is required and separated from the kind by whitespace, and the arrow has to
# be the one that kind uses. Written `\s*` with an optional name, the pattern read
# `[stricttest: a => b]` as a strict rule named "test", accepted a rule with no name at
# all under an invented one, and took either arrow for either kind (#19).
_RULE = re.compile(r"^\[(defeasible|strict)\s+([A-Za-z]\w*)\s*:\s*(.+?)\s*(=>|->)\s*"
                   r"(-?[A-Za-z]\w*)\]$")
ARROW = {"defeasible": "=>", "strict": "->"}
_PREF = re.compile(r"^\[prefer_(rule|premise)\s*:\s*(-?[A-Za-z]\w*)\s*>\s*(-?[A-Za-z]\w*)\]$")


@dataclass
class ParsedAnswer:
    ops: List[Operation] = field(default_factory=list)
    n_lines: int = 0
    n_unparseable: int = 0
    unparseable_examples: List[str] = field(default_factory=list)


def parse_answer(text: str) -> ParsedAnswer:
    """The directives a submission carries, wrapped in delimiters or not.

    `extract_answer` unwraps a wrapped answer and hands back anything else whole, so a
    bare directive list and the same list between [answer] and [/answer] parse to the
    same operations (`docs/dataset-contract.md` section 1).
    """
    out = ParsedAnswer()
    body = extract_answer(text)
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


def score_item(answer_text: str, item: Dict, strict_parse: bool = True) -> ScoreResult:
    """Score one construction answer.

    `success` is the contract's definition for these six tasks -- every goal met and the
    theory left consistent -- and it is returned on every path. An answer that never
    reaches the goal check has not met the goals, so its `success` is False; returning
    nothing there is what made `success_rate` an average over the rows that happened to
    reach a late branch (`docs/dataset-contract.md` section 5).
    """
    base_ops = item["base_ops"]
    ordering = item["ordering"]
    goals: List[Dict] = item["goals"]
    minimum = item.get("min_directives")

    diag: Dict = {"n_lines": 0, "n_unparseable": 0, "illegal": [], "rejected_detail": [],
                  "goals_met": [], "n_used": 0, "minimum": minimum}

    def failed(reason: str, score: float = 0.0) -> ScoreResult:
        return ScoreResult(score=score, success=False, reason=reason, diagnostics=diag)

    p = parse_answer(answer_text)
    diag.update(n_lines=p.n_lines, n_unparseable=p.n_unparseable,
                unparseable_examples=p.unparseable_examples)
    if strict_parse and p.n_unparseable:
        return failed(f"unparseable_lines:{p.n_unparseable}")
    if not p.ops:
        # An empty submission lands here too. It is an answer with nothing in it, not an
        # answer that failed to arrive: delimiters are delivery and the evaluator owns
        # them, so there is no such outcome as a missing answer region.
        return failed("no_directives")

    kept, illegal = check_legality(p.ops, base_ops, item.get("allow_strict", False),
                                   item.get("preferences_only", False))
    diag["illegal"] = illegal
    n_used = p.n_lines
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
    diag["deadlock_not_defeat"] = sum(
        1 for g in diag["goals_met"]
        if g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED"))

    if not success:
        subgoals = item.get("subgoals") or []
        sub_progress = None
        if subgoals:
            hit = 0
            for lit in subgoals:
                try:
                    if str(v.status(lit)) != "JUSTIFIED":
                        hit += 1
                except Exception:
                    pass
            sub_progress = hit / len(subgoals)
            diag["subgoals_defeated"] = f"{hit}/{len(subgoals)}"

        per_goal = []
        for g in diag["goals_met"]:
            if g["got"] == g["want"]:
                per_goal.append(1.0)
            elif g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED"):
                per_goal.append(0.4)
            elif sub_progress is not None and g["want"] == "OVERRULED":
                per_goal.append(0.3 * sub_progress)
            else:
                per_goal.append(0.0)
        progress = (sum(per_goal) / len(per_goal)) if per_goal else 0.0
        diag["progress"] = round(progress, 4)
        partial = round(PARTIAL_CAP * progress, 4) if consistent else 0.0
        return failed("goal_not_met", score=partial)
    if not minimum:
        # Every goal met, so the task is done; the minimum is unknown, so economy cannot
        # be measured. Success without the efficiency half of the score.
        return ScoreResult(score=0.5, success=True, reason="success_but_minimum_unknown",
                           diagnostics=diag)
    efficiency = min(1.0, minimum / max(n_used, 1))
    diag["efficiency"] = round(efficiency, 4)
    return ScoreResult(score=round(0.5 + 0.5 * efficiency, 4), success=True, reason="ok",
                       diagnostics=diag)
