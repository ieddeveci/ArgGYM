from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier
from arggym.core.curriculum import junction_budget, JUNCTION_CAPS, PROFILES, junctions_for
from arggym.core.curriculum import negated_branch
from arggym.core.invariants import (randomize_rule_names, language_enrichment,
                            minimal_subset_exact)

TASK = "preference_construction"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
EASY_LEVELS = 4
_L = "abcdefghijklmnopqrstuvwxy"


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def _names(seed: int, n: int) -> List[str]:
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:12] for b in _L[12:] for d in range(10)]
    rng.shuffle(pool)
    return pool[:n]


def _ordered(ops: Sequence[Operation], shuffle_seed: Optional[int] = None) -> List[Operation]:
    facts = [o for o in ops if o.kind in ("premise", "axiom")]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    prefs = [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(facts)
        rng.shuffle(rules)
    return facts + rules + prefs


def render_ops(ops: Sequence[Operation]) -> str:
    out = []
    for o in ops:
        if o.kind in ("premise", "axiom"):
            out.append(f"[{o.kind}: {o.content}]")
        elif o.kind in ("defeasible", "strict"):
            arrow = "=>" if o.kind == "defeasible" else "->"
            out.append(f"[{o.kind} {o.name}: {' AND '.join(o.antecedents)} {arrow} {o.consequent}]")
        else:
            out.append(f"[{o.kind}: {o.stronger} > {o.weaker}]")
    return "\n".join(out)


def status(ops: Sequence[Operation], lit: str, ordering: str) -> str:
    try:
        return str(ASPICVerifier.from_operations(list(ops), ordering=ordering).status(lit))
    except Exception:
        return "ERR"


def statuses(ops: Sequence[Operation], lits: Sequence[str], ordering: str) -> Dict[str, str]:
    try:
        v = ASPICVerifier.from_operations(list(ops), ordering=ordering)
        return {l: str(v.status(l)) for l in lits}
    except Exception:
        return {}


@dataclass
class PCItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    goals: List[Dict]
    ordering: str
    level: int
    reference: str
    min_directives: int
    metadata: Dict = field(default_factory=dict)

    def as_score_input(self) -> Dict:
        return {"base_ops": self.base_ops, "ordering": self.ordering,
                "goals": [{"claim": g["claim"], "want": g["want"]} for g in self.goals],
                "min_directives": self.min_directives,
                "allow_strict": False, "preferences_only": True}


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Optional[PCItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "pc"))
    n_claims = max(1, min(1 + (level * 8) // 15, 8))
    n_conf = max(n_claims + 1, min(n_claims + 1 + (level * 5) // 15, 10))
    depth = max(1, min(1 + (level * 4) // 15, 5))
    if level >= 3:
        n_conf = min(10, n_conf + rng.randint(0, 1))
        depth = max(1, min(5, depth + rng.randint(-1 if level >= 8 else 0, 1)))
    shared = level >= 5 and (level % 3 == 2)
    if shared:
        n_claims = min(8, n_claims + 1)
        n_conf = min(10, max(n_conf, n_claims + 1))
    with_prefs = level >= 10

    names = _names(stable_seed(seed, level, ordering, "nm"), 20 + n_conf * (depth + 6))
    it = iter(names)
    ops: List[Operation] = []
    conflicts: List[Dict] = []
    ridx = 0

    j_budget = junctions_for(level, max(1, n_conf * (depth + 2) + 4))
    for ci in range(n_conf):
        pro_root, con_root = next(it), next(it)
        ops.append(Operation(kind="premise", content=pro_root))
        ops.append(Operation(kind="premise", content=con_root))
        node = next(it)
        cur = pro_root
        pro_rules: List[str] = []
        _pts = set()
        if depth >= 2 and ci < j_budget:
            _pts.add(min(depth // 2, depth - 1))
            if j_budget > n_conf and depth >= 3:
                _pts.add(0)
        junction_at = -1
        for j in range(depth):
            ridx += 1
            nm = f"d{ridx}"
            nxt = node if j == depth - 1 else next(it)
            if j in _pts:
                broot = next(it)
                _bsrc = ("-" + broot) if negated_branch(ci) else broot
                ops.append(Operation(kind="premise", content=_bsrc))
                ridx += 1
                bnm = f"d{ridx}"
                blit = next(it)
                ops.append(Operation(kind="defeasible", name=bnm, antecedents=(_bsrc,),
                                     consequent=blit))
                pro_rules.append(bnm)
                ridx += 1
                nm = f"d{ridx}"
                ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur, blit),
                                     consequent=nxt))
            else:
                ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,),
                                     consequent=nxt))
            pro_rules.append(nm)
            cur = nxt
        ridx += 1
        con_rule = f"d{ridx}"
        ops.append(Operation(kind="defeasible", name=con_rule, antecedents=(con_root,),
                             consequent="-" + node))
        conflicts.append({"node": node, "pro": pro_rules, "con": con_rule,
                          "pro_root": pro_root, "con_root": con_root, "claims": []})

    claim_lits: List[str] = []
    for k in range(n_claims):
        if shared and k <= 1:
            src = conflicts[0]
        elif k < n_conf:
            src = conflicts[k]
        else:
            src = conflicts[k % n_conf]
        lit = next(it)
        ridx += 1
        ops.append(Operation(kind="defeasible", name=f"d{ridx}", antecedents=(src["node"],),
                             consequent=lit))
        src["claims"].append(lit)
        claim_lits.append(lit)

    resolved: List[Dict] = []
    if with_prefs:
        for c in conflicts[:max(1, n_conf // 3)]:
            ops.append(Operation(kind="prefer_rule", stronger=c["pro"][-1], weaker=c["con"]))
            resolved.append(c)

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    ops = list(ops) + _lx
    ops, _rmap = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    for _c in conflicts:
        _c["pro"] = [_rmap.get(x, x) for x in _c["pro"]]
        _c["con"] = _rmap.get(_c["con"], _c["con"])
    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))

    atoms = {a.lstrip("-") for o in base for a in
             (list(o.antecedents or ()) + ([o.consequent] if o.consequent else [])
              + ([o.content] if o.content else []))}
    rnames = {o.name for o in base if o.kind in ("defeasible", "strict") and o.name}
    if atoms & rnames:
        return None

    before = statuses(base, claim_lits, ordering)
    if not before:
        return None

    cand: List[Operation] = []
    for c in conflicts:
        if not c["claims"]:
            continue
        want_pro = rng.random() < 0.5
        if c in resolved and rng.random() < 0.5:
            cand.append(Operation(kind="prefer_rule", stronger=c["con"], weaker=c["pro"][-1]))
        elif want_pro:
            cand.append(Operation(kind="prefer_rule", stronger=c["pro"][-1], weaker=c["con"]))
            if ordering == WEAKEST_LINK:
                for r in c["pro"]:
                    cand.append(Operation(kind="prefer_rule", stronger=r, weaker=c["con"]))
                cand.append(Operation(kind="prefer_premise", stronger=c["pro_root"],
                                      weaker=c["con_root"]))
        else:
            cand.append(Operation(kind="prefer_rule", stronger=c["con"], weaker=c["pro"][-1]))
            if ordering == WEAKEST_LINK:
                cand.append(Operation(kind="prefer_premise", stronger=c["con_root"],
                                      weaker=c["pro_root"]))
    if not cand:
        return None
    after = statuses(base + cand, claim_lits, ordering)
    if not after:
        return None
    goals = [{"claim": l, "current": before[l], "want": after[l]} for l in claim_lits]
    if all(g["current"] == g["want"] for g in goals):
        return None

    def holds(sub: Sequence[Operation]) -> bool:
        got = statuses(base + list(sub), claim_lits, ordering)
        return all(got.get(g["claim"]) == g["want"] for g in goals)

    if not holds(cand):
        return None
    best, proven, n_calls = minimal_subset_exact(cand, holds, max_calls=40000)
    if best is None:
        return None

    if len(best) > 1:
        singles: List[Operation] = []
        for c in conflicts:
            if not c["claims"]:
                continue
            singles.append(Operation(kind="prefer_rule", stronger=c["pro"][-1], weaker=c["con"]))
            singles.append(Operation(kind="prefer_rule", stronger=c["con"], weaker=c["pro"][-1]))
            singles.append(Operation(kind="prefer_premise", stronger=c["pro_root"],
                                     weaker=c["con_root"]))
            singles.append(Operation(kind="prefer_premise", stronger=c["con_root"],
                                     weaker=c["pro_root"]))
        found = None
        for cand in singles:
            if holds([cand]):
                found = [cand]
                break
        PAIR_BUDGET = max(60, 240 // max(1, len(base) // 25))
        if found is None and len(best) > 2 and len(singles) * (len(singles) - 1) // 2 <= PAIR_BUDGET:
            for i in range(len(singles)):
                for j in range(i + 1, len(singles)):
                    if holds([singles[i], singles[j]]):
                        found = [singles[i], singles[j]]
                        break
                if found:
                    break
        if found is not None and len(found) < len(best):
            best = found

    lines = [f"[{o.kind}: {o.stronger} > {o.weaker}]" for o in best]
    prompt = _render_prompt(render_ops(base), goals, ordering)
    return PCItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, goals=goals,
        ordering=ordering, level=level,
        reference="[answer]\n" + "\n".join(lines) + "\n[/answer]",
        min_directives=len(best),
        metadata={
            "n_claims": n_claims, "n_conflicts": n_conf, "chain_depth": depth,
            "shared_conflict": shared, "theory_has_preferences": with_prefs,
            "n_theory_preferences": len(resolved),
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
            "min_within_reference": len(best),
            "minimality_proven": proven, "minimality_calls": n_calls,
            "wanted": [g["want"] for g in goals],
        })


_STATUS_WORD = {"JUSTIFIED": "justified", "OVERRULED": "overruled", "UNDECIDED": "undecided"}


def _render_prompt(theory: str, goals: Sequence[Dict], ordering: str) -> str:
    on = "the last-link strength ordering" if ordering == LAST_LINK \
        else "the weakest-link strength ordering"
    lines = [f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
             f"with {on}.", "", theory, ""]
    for g in goals:
        lines.append(f"The claim {g['claim']} is currently "
                     f"{_STATUS_WORD.get(g['current'], g['current'].lower())}.")
    wants = ", ".join(f"{g['claim']} {_STATUS_WORD.get(g['want'], g['want'].lower())}"
                      for g in goals)
    verb = "makes" if len(goals) == 1 else "simultaneously makes"
    lines += ["", f"What is the minimal set of preference directives that {verb} {wants}?", "",
              "Permitted additions: preference directives only. "
              "No new rules or premises may be added.", "",
              "Answer format: one directive per line, between [answer] and [/answer]."]
    return "\n".join(lines)


def make_item(level: int, seed: int, ordering: str = LAST_LINK, profile: str = "FULL",
              tries: int = 14) -> Optional[PCItem]:
    for k in range(tries):
        it = build(level, seed * 61 + k, ordering, profile)
        if it is not None:
            return it
    return None
