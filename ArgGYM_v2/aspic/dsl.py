from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from aspic.engine import Operation

_BRACKET_RE = re.compile(r"\[\s*([a-zA-Z_-]+)(?:\s+([a-zA-Z]\w*))?\s*(?::\s*(.*?))?\s*\]", re.DOTALL)
_AND_RE = re.compile(r"\s+AND\s+")

_SANITIZE_RE = re.compile(r"[\[\]:>]")

def sanitize_statement(s: str) -> str:
    s = (s or "").replace("=>", " ").replace("->", " ")
    s = _SANITIZE_RE.sub(" ", s)
    s = re.sub(r"\s+AND\s+", " and ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".;, ").strip()

_KEYWORDS = {
    "premise": "premise",
    "axiom": "axiom",
    "defeasible": "defeasible",
    "strict": "strict",
    "prefer_rule": "prefer_rule",
    "prefer_premise": "prefer_premise",
}

@dataclass
class ParseResult:
    operations: List[Operation] = field(default_factory=list)
    dropped: List[Dict[str, str]] = field(default_factory=list)

class _Drop(Exception):
    pass

def _antecedents(lhs: str) -> List[str]:
    parts = [p.strip() for p in _AND_RE.split(lhs)]
    parts = [p for p in parts if p]
    if not parts:
        raise _Drop("empty_antecedents")
    return parts

def _parse_one(keyword: str, label: str, body: str, counters: Dict[str, int]) -> Operation:
    kind = _KEYWORDS.get(keyword)
    if kind is None:
        raise _Drop(f"unknown_keyword:{keyword}")

    if kind in ("premise", "axiom"):
        if not body:
            raise _Drop(f"empty_{kind}")
        return Operation(kind=kind, content=body)

    if kind == "defeasible":
        if "=>" not in body:
            raise _Drop("defeasible_missing_arrow")
        lhs, rhs = body.split("=>", 1)
        cons = rhs.strip()
        if not cons:
            raise _Drop("empty_consequent")
        counters["d"] += 1
        return Operation(kind="defeasible", name=label or f"d{counters['d']}",
                         antecedents=tuple(_antecedents(lhs)), consequent=cons)

    if kind == "strict":
        if "->" not in body:
            raise _Drop("strict_missing_arrow")
        lhs, rhs = body.split("->", 1)
        cons = rhs.strip()
        if not cons:
            raise _Drop("empty_consequent")
        counters["s"] += 1
        return Operation(kind="strict", name=label or f"s{counters['s']}",
                         antecedents=tuple(_antecedents(lhs)), consequent=cons)

    if body.count(">") != 1:
        raise _Drop("preference_needs_single_gt")
    left, right = body.split(">", 1)
    stronger, weaker = left.strip(), right.strip()
    if not stronger or not weaker:
        raise _Drop("preference_empty_side")
    return Operation(kind=kind, stronger=stronger, weaker=weaker)

def parse_dsl(text: str, counters: Optional[Dict[str, int]] = None) -> ParseResult:
    if counters is None:
        counters = {"d": 0, "s": 0}
    res = ParseResult()
    for m in _BRACKET_RE.finditer(text or ""):
        keyword = m.group(1).strip().lower()
        label = (m.group(2) or "").strip() or None
        body = (m.group(3) or "").strip()
        try:
            res.operations.append(_parse_one(keyword, label, body, counters))
        except _Drop as d:
            res.dropped.append({"raw": m.group(0), "reason": str(d)})
    return res
