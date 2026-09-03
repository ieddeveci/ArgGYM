from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from arggym.core.pairs import collect, pair_f1

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier
from arggym.core.curriculum import junction_budget, JUNCTION_CAPS, PROFILES, junctions_for
from arggym.core.curriculum import negated_branch
from arggym.core.invariants import randomize_rule_names, language_enrichment

TASK = "perturbation"
HELD_FRAC = 0.35
MAX_STATUS_SHARE = 0.45


def _required_statuses(n_pert: int) -> int:
    return max(1, min(3, n_pert))


EASY_LEVELS = 4


def _max_share(required: int) -> float:
    return {1: 1.0, 2: 0.75}.get(required, MAX_STATUS_SHARE)

PANEL_THRESHOLD = round(MAX_STATUS_SHARE + 0.03, 2)
REQUIRE_ALL_THREE = True
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


def status_map(ops: Sequence[Operation], ordering: str) -> Dict[str, str]:
    try:
        v = ASPICVerifier.from_operations(list(ops), ordering=ordering)
        return {k: str(x) for k, x in v.status_map().items()}
    except Exception:
        return {}


@dataclass
class PerturbItem:
    prompt: str
    theory_text: str
    perturbation_text: str
    base_ops: List[Operation]
    pert_ops: List[Operation]
    ordering: str
    level: int
    gold: Dict[str, str]
    before: Dict[str, str]
    after: Dict[str, str]
    survivors: List[str]
    metadata: Dict = field(default_factory=dict)

    def reference(self) -> str:
        if not self.gold:
            return "[answer]\nnone\n[/answer]"
        return ("[answer]\n"
                + "\n".join(f"{k}: {v.lower()}" for k, v in sorted(self.gold.items()))
                + "\n[/answer]")


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Optional[PerturbItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "pert"))
    n_comp = max(2, min(2 + (level * 7 + 5) // 15, 9))
    depth = max(2, min(2 + (level * 5) // 15, 7))
    n_pert = max(1, min(1 + (level * 9) // 15, 10))
    if level > EASY_LEVELS:
        n_pert = max(n_pert, n_comp)
    lo_comp = max(2, 2 + level // 2 - 1)
    lo_depth = max(2, 2 + level // 3 - (1 if level >= 6 else 0))
    n_comp = max(lo_comp, min(n_comp, 9))
    depth = max(lo_depth, min(depth, 7))
    n_alt = max(1, n_comp // 3)

    names = _names(stable_seed(seed, level, ordering, "nm"), (8 + n_comp * (depth + 4) * 3))
    it = iter(names)
    src = next(it)
    ops: List[Operation] = [Operation(kind="premise", content=src)]
    comps: List[Dict] = []
    ridx = 0

    n_axiom = max(0, n_comp // 4)
    j_budget = junctions_for(level, max(1, n_comp * depth + n_pert + 6))
    for ci in range(n_comp):
        root = next(it)
        is_ax = ci < n_axiom
        ops.append(Operation(
            kind="axiom" if (is_ax and PROFILES[profile].permits("axiom")) else "premise",
            content=root))
        cur = root
        chain: List[Tuple[str, str, bool]] = []
        strict_at = ((depth // 2) if (depth >= 3 and ci % 3 == 0
                                     and PROFILES[profile].permits("strict")) else -1)
        _pts = set()
        if depth >= 2 and ci < j_budget:
            _pts.add(min(depth - 1, depth // 2 + 1))
            if j_budget > n_comp and depth >= 4:
                _pts.add(max(1, depth // 3))
        junction_at = -1
        if junction_at == strict_at:
            junction_at = -1
        for j in range(depth):
            ridx += 1
            nm = f"d{ridx}"
            nxt = next(it)
            strict = (j == strict_at)
            if j in _pts:
                broot = next(it)
                _bsrc = ("-" + broot) if negated_branch(ci) else broot
                ops.append(Operation(kind="premise", content=_bsrc))
                ridx += 1
                bnm = f"d{ridx}"
                blit = next(it)
                ops.append(Operation(kind="defeasible", name=bnm, antecedents=(_bsrc,),
                                     consequent=blit))
                chain.append((bnm, blit, False))
                ridx += 1
                nm = f"d{ridx}"
                ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur, blit),
                                     consequent=nxt))
            else:
                ops.append(Operation(kind="strict" if strict else "defeasible", name=nm,
                                     antecedents=(cur,), consequent=nxt))
            chain.append((nm, nxt, strict))
            cur = nxt
        comps.append({"root": root, "chain": chain, "alt": False, "held": False,
                      "axiom": is_ax, "strict_at": strict_at})

    cq = next(it)
    ops.append(Operation(kind="premise", content=cq))
    ops.append(Operation(kind="premise", content="-" + cq))
    ridx += 1
    cq_rule = f"d{ridx}"
    cq_lit = next(it)
    ops.append(Operation(kind="defeasible", name=cq_rule, antecedents=(cq,), consequent=cq_lit))

    n_held = max(1, int(n_comp * HELD_FRAC))
    for ci in range(n_comp - n_held, n_comp):
        c = comps[ci]
        tgt_idx = max(0, len(c["chain"]) // 2 - 1)
        tgt_lit = c["chain"][tgt_idx][1]
        atk_root = next(it)
        ops.append(Operation(kind="premise", content=atk_root))
        ridx += 1
        atk_rule = f"d{ridx}"
        ops.append(Operation(kind="defeasible", name=atk_rule, antecedents=(atk_root,),
                             consequent="-" + tgt_lit))
        c["held"] = True
        c["attacker"] = atk_rule
        c["held_lit"] = tgt_lit
        if ci % 2 == 0:
            sup_rule = c["chain"][tgt_idx][0]
            if not c["chain"][tgt_idx][2]:
                ops.append(Operation(kind="prefer_rule", stronger=atk_rule, weaker=sup_rule))
                c["held_by_pref"] = (atk_rule, sup_rule)

    for ci in range(min(n_alt, n_comp)):
        c = comps[ci]
        if len(c["chain"]) < 3:
            continue
        mid_idx = len(c["chain"]) // 2 + 1
        target_lit = c["chain"][mid_idx][1]
        alt_root = next(it)
        ops.append(Operation(kind="premise", content=alt_root))
        ridx += 1
        ops.append(Operation(kind="defeasible", name=f"d{ridx}", antecedents=(alt_root,),
                             consequent=target_lit))
        c["alt"] = True
        c["alt_from"] = mid_idx

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    ops = list(ops) + _lx
    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))

    pert: List[Operation] = []
    order = list(range(n_comp))
    rng.shuffle(order)
    pidx = 0
    for k in range(n_pert):
        ci = order[k % n_comp]
        c = comps[ci]
        if c["held"]:
            pidx += 1
            hp = c.get("held_by_pref")
            if hp and k % 2 == 1:
                atk_rule, sup_rule = hp
                pert.append(Operation(kind="prefer_rule", stronger=sup_rule, weaker=atk_rule))
            else:
                pert.append(Operation(kind="defeasible", name=f"k{pidx}", antecedents=(src,),
                                      consequent="-" + c["attacker"]))
            continue
        cut_at = 0 if not c["alt"] else max(0, c.get("alt_from", 1) - 1)
        rule_name, cons, is_strict = c["chain"][min(cut_at, len(c["chain"]) - 1)]
        pidx += 1
        if is_strict:
            pert.append(Operation(kind="defeasible", name=f"k{pidx}", antecedents=(src,),
                                  consequent="-" + rule_name))
            continue
        if k % 2 == 0:
            pert.append(Operation(kind="defeasible", name=f"k{pidx}", antecedents=(src,),
                                  consequent="-" + rule_name))
        else:
            pert.append(Operation(kind="defeasible", name=f"k{pidx}", antecedents=(src,),
                                  consequent="-" + cons))

    if n_pert >= 2:
        pert.append(Operation(kind="prefer_premise", stronger=cq, weaker="-" + cq))

    _all, _rmap = randomize_rule_names(base + pert, stable_seed(seed, level, ordering, "rn"))
    base, pert = _all[:len(base)], _all[len(base):]
    for _c in comps:
        _c["chain"] = [(_rmap.get(a, a), b, c2) for (a, b, c2) in _c["chain"]]
        if _c.get("attacker"):
            _c["attacker"] = _rmap.get(_c["attacker"], _c["attacker"])
        if _c.get("held_by_pref"):
            _a, _b = _c["held_by_pref"]
            _c["held_by_pref"] = (_rmap.get(_a, _a), _rmap.get(_b, _b))
    cq_rule = _rmap.get(cq_rule, cq_rule)

    rules = {o.name for o in base + pert
             if o.kind in ("defeasible", "strict") and o.name}
    atoms = set()
    for o in base + pert:
        if o.content:
            atoms.add(o.content.lstrip("-"))
        for a in (list(o.antecedents or ()) + ([o.consequent] if o.consequent else [])):
            if a.startswith("-") and a[1:] in rules:
                continue
            atoms.add(a.lstrip("-"))
    if atoms & rules:
        return None

    before = status_map(base, ordering)
    if not before:
        return None

    after = status_map(base + pert, ordering)
    if not after:
        return None

    changed = {k: after[k] for k in before
               if k in after and after[k] != before[k]}
    changed.update({k: after[k] for k in after if k not in before})
    changed = {k: v for k, v in changed.items() if k in before}

    downstream: Set[str] = set()
    for c in comps:
        for _nm, cons, _st in c["chain"]:
            downstream.add(cons)
    survivors = sorted(x for x in downstream if x not in changed and x in before)
    survivors += sorted(f"-{x}" for x in downstream
                        if f"-{x}" in before and f"-{x}" not in changed)

    if not changed:
        return None
    if not survivors:
        return None

    counts: Dict[str, int] = {}
    for v in changed.values():
        counts[v] = counts.get(v, 0) + 1
    need = _required_statuses(n_pert)
    if len(counts) < need:
        return None
    if max(counts.values()) / len(changed) > _max_share(need):
        return None

    prompt = _render_prompt(render_ops(base), render_ops(pert), ordering)
    n_status = len(set(changed.values()))
    return PerturbItem(
        prompt=prompt, theory_text=render_ops(base), perturbation_text=render_ops(pert),
        base_ops=base, pert_ops=pert, ordering=ordering, level=level,
        gold=changed, before=before, after=after, survivors=survivors,
        metadata={
            "n_components": n_comp, "chain_depth": depth, "n_perturbations": n_pert,
            "n_alt_support": sum(1 for c in comps if c["alt"]),
            "n_changed": len(changed), "n_survivors": len(survivors),
            "n_distinct_new_statuses": n_status,
            "new_statuses": sorted(set(changed.values())),
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
            "n_atoms": len(atoms),
            "changed_fraction": round(len(changed) / max(len(before), 1), 3),
        })


# A perturbation may declare a preference the theory already declares the other way.
# The DSL has no removal, so that is the only way to write "this preference no longer
# decides the conflict", and 13 of the 40 exported items use it. The engine reads the
# pair as equally preferred and gold follows the engine, so the item is sound -- but the
# model was being graded on a convention no prompt stated (#6).
TIE_NOTE = ("Preference is a preorder, so a pair declared stronger in both directions is "
            "equally preferred and settles nothing between them.")


def _render_prompt(theory: str, pert: str, ordering: str) -> str:
    on = _ordering_phrase(ordering)
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"The following directives are then added to the theory:\n\n{pert}\n\n"
            "Which claims of the original theory change status, and what does each new status become?\n"
            "Claims include negated literals such as -x, where those appear in the theory.\n"
            "Possible statuses: justified, overruled, undecided.\n"
            "A claim is justified when some argument for it is accepted, overruled when "
            "every argument for it is defeated, and undecided otherwise.\n"
            f"{TIE_NOTE}\n\n"
            "Answer format: one line per changed claim, written as `claim: status`, between [answer] "
            "and [/answer]. If no claim changes status, write `none`.")


_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)
_PAIR = re.compile(r"(-?\w+)\s*[:=]\s*(justified|overruled|undecided)\b", re.I)


def score(answer_text: str, item: PerturbItem, strict_parse: bool = True) -> Dict:
    diag: Dict = {"n_lines": 0, "n_unparseable": 0, "n_predicted": 0,
                  "n_gold": len(item.gold), "wrong_status": [], "false_positives": [],
                  "missed": [], "contradicted": []}
    m = _ANSWER.search(answer_text or "")
    if m is None:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    body = m.group(1).strip()
    if body.lower() == "none":
        diag["n_lines"] = 1
        return {"score": 0.0, "reason": "predicted_none", "f1": 0.0,
                "exact_match": False, "diagnostics": diag}
    matches = _PAIR.findall(body)
    diag["n_lines"] = len(matches)
    pred = collect((claim.lower(), stat.upper()) for claim, stat in matches)
    residue = _PAIR.sub(" ", body)
    for tok in residue.split():
        if tok.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", tok):
            diag["n_unparseable"] += 1
    if strict_parse and diag["n_unparseable"]:
        return {"score": 0.0, "reason": f"unparseable_lines:{diag['n_unparseable']}",
                "diagnostics": diag}
    if not pred:
        return {"score": 0.0, "reason": "no_pairs", "diagnostics": diag}
    diag["n_predicted"] = len(pred)

    gold = item.gold
    r = pair_f1(pred, gold)
    diag["wrong_status"] = sorted(k for k, v in pred.items() if k in gold and v != [gold[k]])
    diag["contradicted"] = r.contradicted
    diag["false_positives"] = sorted(k for k in pred if k not in gold)
    diag["missed"] = sorted(k for k in gold if k not in pred)
    surv = set(item.survivors)
    diag["survivor_included"] = sorted(k for k in diag["false_positives"] if k in surv)
    diag["spurious"] = sorted(k for k in diag["false_positives"] if k not in surv)
    diag["n_survivor_included"] = len(diag["survivor_included"])
    diag["n_wrong_status"] = len(diag["wrong_status"])
    diag["n_contradicted"] = len(diag["contradicted"])
    diag["n_missed"] = len(diag["missed"])
    diag["n_spurious"] = len(diag["spurious"])
    return {"score": round(r.f1, 4), "reason": "ok", "f1": round(r.f1, 4),
            "precision": round(r.precision, 4), "recall": round(r.recall, 4),
            "exact_match": r.exact_match, "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK, profile: str = "FULL",
              tries: int = 40) -> Optional[PerturbItem]:
    for k in range(tries):
        it = build(level, seed * 41 + k, ordering, profile)
        if it is not None:
            return it
    return None
