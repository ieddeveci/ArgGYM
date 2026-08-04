from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier

_ANSWER_OPEN = re.compile(r"\[\s*answer\s*\]", re.I)
_ANSWER_CLOSE = re.compile(r"\[\s*/\s*answer\s*\]", re.I)


def answer_region(text: Optional[str]) -> Optional[str]:
    """The content of the FINAL complete [answer]...[/answer] region.

    The last region, not the first: a reasoning model routinely drafts an answer
    mid-chain-of-thought and then revises it, and what it submits is the
    revision. Scoring the abandoned draft would report the model as wrong on
    work it had already corrected. Returns None when no complete region exists,
    which the callers report as a format failure rather than a wrong answer.
    """
    s = text or ""
    for m in reversed(list(_ANSWER_OPEN.finditer(s))):
        rest = s[m.end():]
        c = _ANSWER_CLOSE.search(rest)
        if c:
            return rest[:c.start()]
    return None


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
    body = answer_region(text)
    if body is None:
        out.no_region = True
        return out
    auto = 0
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
    ordinary = {o.content for o in base_ops if o.kind == "premise"}
    axioms = {o.content for o in base_ops if o.kind == "axiom"}
    kept: List[Operation] = []
    reasons: List[str] = []
    for o in ops:
        if o.kind == "strict" and not allow_strict:
            reasons.append(f"illegal_strict_rule:{o.name or '?'}")
            continue
        if prefs_only and o.kind not in ("prefer_rule", "prefer_premise"):
            reasons.append(f"illegal_non_preference:{o.kind}")
            continue
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


def _apply(base_ops: Sequence[Operation], added: Sequence[Operation]):
    plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
    prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
    return list(base_ops) + plain + prefs


def score_item(answer_text: str, item: Dict, strict_parse: bool = True) -> Dict:
    base_ops = item["base_ops"]
    ordering = item["ordering"]
    goals: List[Dict] = item["goals"]
    minimum = item.get("min_directives")

    diag: Dict = {"n_lines": 0, "n_unparseable": 0, "illegal": [], "goals_met": [],
                  "n_used": 0, "minimum": minimum}

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
    n_used = len(kept)
    diag["n_used"] = n_used

    diag["achieved_status"] = {g["claim"]: g["got"] for g in diag["goals_met"]}
    diag["deadlock_not_defeat"] = sum(
        1 for g in diag["goals_met"]
        if g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED"))

    if not success:
        return {"score": 0.0, "reason": "goal_not_met", "success": False,
                "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"],
                "diagnostics": diag}
    if not minimum:
        return {"score": 0.5, "reason": "success_but_minimum_unknown", "success": True,
                "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"],
                "diagnostics": diag}
    efficiency = min(1.0, minimum / max(n_used, 1))
    score = round(0.5 + 0.5 * efficiency, 4)
    return {"score": score, "reason": "ok", "success": True,
            "efficiency": round(efficiency, 4),
            "achieved_status": diag["achieved_status"],
            "deadlock_not_defeat": diag["deadlock_not_defeat"],
            "diagnostics": diag}
