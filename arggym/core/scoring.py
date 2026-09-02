from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier

BLOAT_FACTOR = 2
PARTIAL_CAP = 0.25

_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)
_PREMISE = re.compile(r"^\[(premise|axiom)\s*:\s*(-?[A-Za-z]\w*)\]$")
_RULE = re.compile(r"^\[(defeasible|strict)\s*([A-Za-z]\w*)?\s*:\s*(.+?)\s*(=>|->)\s*(-?[A-Za-z]\w*)\]$")
_PREF = re.compile(r"^\[prefer_(rule|premise)\s*:\s*(-?[A-Za-z]\w*)\s*>\s*(-?[A-Za-z]\w*)\]$")


@dataclass
class ParsedAnswer:
    ops: List[Operation] = field(default_factory=list)
    n_lines: int = 0
    n_unparseable: int = 0
    unparseable_examples: List[str] = field(default_factory=list)
    no_region: bool = False


def parse_answer(text: str) -> ParsedAnswer:
    out = ParsedAnswer()
    m = _ANSWER.search(text or "")
    if m is None:
        out.no_region = True
        return out
    auto = 0
    body = m.group(1)
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
            auto += 1
            name = mm.group(2) or f"m{auto}"
            ants = tuple(a.strip() for a in mm.group(3).split("AND") if a.strip())
            if not ants:
                out.n_unparseable += 1
                out.unparseable_examples.append(line[:60])
                continue
            out.ops.append(Operation(kind=mm.group(1), name=name,
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
    """Drop directives the docs forbid or the engine would reject; return (kept, reasons).

    Preferences are checked against the base theory plus the answer's own kept
    rules and premises, with the same conditions as ``ASPICFramework``
    (``engine.py:add_rule_preference`` / ``add_premise_preference``): both names
    must exist, and be distinct. The engine tests existence before identity, so a
    self-preference on an unknown name reports as unknown.
    """
    ordinary = {o.content for o in base_ops if o.kind == "premise"}
    axioms = {o.content for o in base_ops if o.kind == "axiom"}
    defeasible = {o.name for o in base_ops if o.kind == "defeasible"}
    verdict: List[Optional[str]] = []  # None = kept, str = reason; prefs decided later
    for o in ops:
        if o.kind == "strict" and not allow_strict:
            verdict.append(f"illegal_strict_rule:{o.name or '?'}")
            continue
        if prefs_only and o.kind not in ("prefer_rule", "prefer_premise"):
            verdict.append(f"illegal_non_preference:{o.kind}")
            continue
        if o.kind in ("premise", "axiom"):
            c = o.content
            if o.kind == "axiom":
                verdict.append(f"illegal_new_axiom:{c}")
                continue
            if not c.startswith("-"):
                verdict.append(f"illegal_new_premise:{c}")
                continue
            base = c[1:]
            if base in axioms:
                verdict.append(f"illegal_undermine_axiom:{c}")
                continue
            if base not in ordinary:
                verdict.append(f"illegal_asserted_contrary:{c}")
                continue
        verdict.append(None)
    for o, v in zip(ops, verdict):
        if v is None and o.kind == "defeasible":
            defeasible.add(o.name)
        elif v is None and o.kind == "premise":
            ordinary.add(o.content)
    kept: List[Operation] = []
    reasons: List[str] = []
    for o, v in zip(ops, verdict):
        if v is None and o.kind == "prefer_rule":
            if o.stronger not in defeasible or o.weaker not in defeasible:
                v = f"unknown_rule_preference:{o.stronger}>{o.weaker}"
            elif o.stronger == o.weaker:
                v = f"self_preference:{o.stronger}"
        elif v is None and o.kind == "prefer_premise":
            if o.stronger not in ordinary or o.weaker not in ordinary:
                v = f"unknown_premise_preference:{o.stronger}>{o.weaker}"
            elif o.stronger == o.weaker:
                v = f"self_preference:{o.stronger}"
        if v is None:
            kept.append(o)
        else:
            reasons.append(v)
    return kept, reasons


def _apply(base_ops: Sequence[Operation], added: Sequence[Operation]):
    plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
    prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
    return list(base_ops) + plain + prefs


def score_item(answer_text: str, item: Dict, strict_parse: bool = True) -> Dict:
    base_ops = item["base_ops"]
    ordering = item["ordering"]
    goals: List[Dict] = item["goals"]
    minimum = item.get("min_directives")

    diag: Dict = {"n_lines": 0, "n_unparseable": 0, "illegal": [], "n_illegal": 0,
                  "goals_met": [], "n_used": 0, "minimum": minimum}

    p = parse_answer(answer_text)
    diag.update(n_lines=p.n_lines, n_unparseable=p.n_unparseable,
                unparseable_examples=p.unparseable_examples)
    if p.no_region:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    if strict_parse and p.n_unparseable:
        return {"score": 0.0, "reason": f"unparseable_lines:{p.n_unparseable}", "diagnostics": diag}
    if not p.ops:
        return {"score": 0.0, "reason": "no_directives", "diagnostics": diag}

    kept, illegal = check_legality(p.ops, base_ops, item.get("allow_strict", False),
                                   item.get("preferences_only", False))
    diag["illegal"] = illegal
    diag["n_illegal"] = len(illegal)
    # Every directive the answer wrote costs economy, dropped or not (issue #21).
    n_used = len(p.ops)
    diag["n_used"] = n_used
    if not kept:
        return {"score": 0.0, "reason": "all_directives_illegal", "diagnostics": diag}

    try:
        v = ASPICVerifier.from_operations(_apply(base_ops, kept), ordering=ordering)
        consistent = v.is_consistent()
    except Exception as e:
        return {"score": 0.0, "reason": f"engine_rejected:{type(e).__name__}", "diagnostics": diag}

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
        partial = round(PARTIAL_CAP * progress, 4) if consistent else 0.0
        return {"score": partial, "reason": "goal_not_met", "success": False,
                "progress": round(progress, 4),
                "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"],
                "diagnostics": diag}
    if not minimum:
        return {"score": 0.5, "reason": "success_but_minimum_unknown", "success": True,
                "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"],
                "diagnostics": diag}
    if n_used > BLOAT_FACTOR * max(minimum, 1):
        diag["bloat_ratio"] = round(n_used / max(minimum, 1), 2)
        return {"score": 0.0, "reason": f"bloated:{n_used}_used_vs_{minimum}_minimum",
                "success": True, "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"], "diagnostics": diag}
    efficiency = min(1.0, minimum / max(n_used, 1))
    score = round(0.5 + 0.5 * efficiency, 4)
    return {"score": score, "reason": "ok", "success": True,
            "efficiency": round(efficiency, 4),
            "achieved_status": diag["achieved_status"],
            "deadlock_not_defeat": diag["deadlock_not_defeat"],
            "diagnostics": diag}
