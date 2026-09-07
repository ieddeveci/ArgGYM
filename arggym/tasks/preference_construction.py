from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.core.build import BuildReport, Rejected, retry
from arggym.core.curriculum import (
    PROFILES,
    junctions_for,
    negated_branch,
)
from arggym.core.invariants import (
    language_enrichment,
    minimal_subset_exact,
    randomize_rule_names,
    split_atoms_and_rules,
)
from arggym.core.prompting import answer_format, preference_block

TASK = "preference_construction"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"

_ORDERING_NAME = {
    "last_link_elitist": "the last-link elitist strength ordering",
    "last_link_democratic": "the last-link democratic strength ordering",
    "weakest_link_elitist": "the weakest-link elitist strength ordering",
    "weakest_link_democratic": "the weakest-link democratic strength ordering",
}


def _ordering_phrase(ordering: str) -> str:
    return _ORDERING_NAME.get(ordering, str(ordering))


def _is_weakest(ordering: str) -> bool:
    return str(ordering).startswith("weakest_link")


def _is_last(ordering: str) -> bool:
    return str(ordering).startswith("last_link")
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
          profile: str = "FULL") -> Union[PCItem, Rejected]:
    rng = random.Random(stable_seed(seed, level, ordering, "pc"))
    n_claims = max(1, min(1 + (level * 8) // 15, 8))
    n_conf = max(n_claims + 1, min(n_claims + 1 + (level * 5) // 15, 10))
    depth = max(1, min(1 + (level * 4) // 15, 5))
    if level >= 3:
        n_conf = min(10, n_conf + rng.randint(0, 1))
        depth = max(1, min(5, depth + rng.randint(-1 if level >= 8 else 0, 1)))
    # The grid steps by 3, so level is a multiple of 3 at every evaluated level and
    # level % 3 is the constant 0 there. The old `level % 3 == 2` picked levels 5, 8, 11
    # and 14, which the grid never evaluates, so no exported item ever shared a conflict.
    # Divide the step out first, turning level into the index it is meant to be, and the
    # alternation lands on the grid. Shared goes on 9 and 15 rather than 6 and 12 because
    # the extra claim below has to fall where the base ramp is flat: n_claims steps up at
    # 6 and 12, so sharing there makes L6 as wide as L9 and L12 as wide as L15, while
    # sharing at 9 and 15 leaves the goal count strictly increasing across the grid.
    shared = level >= 9 and (level // 3) % 2 == 1
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

    atoms, rnames = split_atoms_and_rules(base)
    if atoms & rnames:
        return Rejected("atom_rule_name_collision")

    before = statuses(base, claim_lits, ordering)
    if not before:
        return Rejected("base_statuses_empty")

    cand: List[Operation] = []
    for c in conflicts:
        if not c["claims"]:
            continue
        want_pro = rng.random() < 0.5
        if c in resolved and rng.random() < 0.5:
            cand.append(Operation(kind="prefer_rule", stronger=c["con"], weaker=c["pro"][-1]))
        elif want_pro:
            cand.append(Operation(kind="prefer_rule", stronger=c["pro"][-1], weaker=c["con"]))
            if _is_weakest(ordering):
                for r in c["pro"]:
                    cand.append(Operation(kind="prefer_rule", stronger=r, weaker=c["con"]))
                cand.append(Operation(kind="prefer_premise", stronger=c["pro_root"],
                                      weaker=c["con_root"]))
        else:
            cand.append(Operation(kind="prefer_rule", stronger=c["con"], weaker=c["pro"][-1]))
            if _is_weakest(ordering):
                cand.append(Operation(kind="prefer_premise", stronger=c["con_root"],
                                      weaker=c["pro_root"]))
    if not cand:
        return Rejected("no_candidate_preferences")
    after = statuses(base + cand, claim_lits, ordering)
    if not after:
        return Rejected("after_statuses_empty")
    goals = [{"claim": l, "current": before[l], "want": after[l]} for l in claim_lits]
    if all(g["current"] == g["want"] for g in goals):
        return Rejected("no_goal_changes")

    def holds(sub: Sequence[Operation]) -> bool:
        got = statuses(base + list(sub), claim_lits, ordering)
        return all(got.get(g["claim"]) == g["want"] for g in goals)

    if not holds(cand):
        return Rejected("candidates_do_not_reach_goals")
    best, proven, n_calls = minimal_subset_exact(cand, holds, max_calls=40000)
    if best is None:
        return Rejected("no_minimal_subset")

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
        reference="\n".join(lines),
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


# The reference answer sometimes declares a preference the theory already declares the
# other way, which is how it undoes a resolved conflict: the DSL has no removal, so a
# reversal is the only available move and the engine reads the pair as equally preferred.
# Eight of the 40 exported items have a gold answer that does this (#6).
TIE_NOTE = ("Preference is a preorder, so a pair declared stronger in both directions is "
            "equally preferred and settles nothing between them.")


def _render_prompt(theory: str, goals: Sequence[Dict], ordering: str) -> str:
    on = _ordering_phrase(ordering)
    lines = [f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
             f"with {on}.", "", theory, ""]
    for g in goals:
        lines.append(f"The claim {g['claim']} is currently "
                     f"{_STATUS_WORD.get(g['current'], g['current'].lower())}.")
    wants = ", ".join(f"{g['claim']} {_STATUS_WORD.get(g['want'], g['want'].lower())}"
                      for g in goals)
    verb = "makes" if len(goals) == 1 else "simultaneously makes"
    lines += ["", f"What is the minimal set of preference directives that {verb} {wants}?", "",
              preference_block(),
              TIE_NOTE, "",
              answer_format("Answer format: one directive per line.")]
    return "\n".join(lines)


def make_item_report(level: int, seed: int, ordering: str = LAST_LINK,
                     profile: str = "FULL", tries: int = 14) -> BuildReport:
    return retry(lambda k: build(level, seed * 61 + k, ordering, profile), tries)


def make_item(level: int, seed: int, ordering: str = LAST_LINK, profile: str = "FULL",
              tries: int = 14) -> Optional[PCItem]:
    return make_item_report(level, seed, ordering, profile, tries).item
