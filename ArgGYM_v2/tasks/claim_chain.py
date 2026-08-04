from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier
from core.invariants import (split_atoms_and_rules, randomize_rule_names, negation_gadget,
                        language_enrichment)

TASK = "claim_chain"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
EASY_LEVELS = 3
_L = "abcdefghijklmnopqrstuvwxy"


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def _names(seed: int, n: int) -> List[str]:
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:14] for b in _L[10:] for d in range(10)]
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


def render_op(o: Operation) -> str:
    if o.kind in ("premise", "axiom"):
        return f"[{o.kind}: {o.content}]"
    if o.kind in ("defeasible", "strict"):
        arrow = "=>" if o.kind == "defeasible" else "->"
        return f"[{o.kind} {o.name}: {' AND '.join(o.antecedents)} {arrow} {o.consequent}]"
    return f"[{o.kind}: {o.stronger} > {o.weaker}]"


def render_ops(ops: Sequence[Operation]) -> str:
    return "\n".join(render_op(o) for o in ops)


def status(ops: Sequence[Operation], lit: str, ordering: str) -> str:
    try:
        return str(ASPICVerifier.from_operations(list(ops), ordering=ordering).status(lit))
    except Exception:
        return "ERR"


@dataclass
class CCItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    claim: str
    line_ops: List[Operation]
    ordering: str
    level: int
    reference: str
    metadata: Dict = field(default_factory=dict)


def _tower(ops: List[Operation], names, ridx: List[int], attacked_lit: str,
           height: int) -> None:
    prev_rule = None
    for i in range(height):
        root = next(names)
        ops.append(Operation(kind="premise", content=root))
        ridx[0] += 1
        nm = f"w_{ridx[0]}"
        cons = ("-" + attacked_lit) if i == 0 else ("-" + prev_rule)
        ops.append(Operation(kind="defeasible", name=nm, antecedents=(root,), consequent=cons))
        prev_rule = nm


def build(level: int, seed: int, ordering: str = LAST_LINK) -> Optional[CCItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "cc"))
    depth = max(2, min(2 + level, 20))
    n_decoy = 1 if level < 4 else min(1 + (level - 4) // 4, 3)
    tower_true = 0 if level < 8 else 2 * min(1 + (level - 8) // 4, 3)
    n_filler = max(0, min(4 + level * 3, 60))
    branch_decoys = level >= 11

    names = _names(stable_seed(seed, level, ordering, "nm"),
                   60 + depth * 3 + n_decoy * (depth + 6) + n_filler * 2 + tower_true * 3)
    it = iter(names)
    claim = next(it)
    ops: List[Operation] = []
    ridx = [0]

    use_neg_root = level >= 4
    if use_neg_root:
        nbase = next(it)
        ops.append(Operation(kind="premise", content=nbase))
        ops.append(Operation(kind="premise", content="-" + nbase))
        ops.append(Operation(kind="prefer_premise", stronger="-" + nbase, weaker=nbase))
        root = "-" + nbase
        line_ops: List[Operation] = [Operation(kind="premise", content=root)]
    else:
        root = next(it)
        ops.append(Operation(kind="premise", content=root))
        line_ops = [ops[-1]]
    cur = root
    mid_lit = None
    for j in range(depth):
        ridx[0] += 1
        nm = f"r_{ridx[0]}"
        nxt = claim if j == depth - 1 else next(it)
        r = Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt)
        ops.append(r)
        line_ops.append(r)
        if j == depth // 2:
            mid_lit = nxt
        cur = nxt
    if tower_true and mid_lit:
        _tower(ops, it, ridx, mid_lit, tower_true)

    decoy_info = []
    for k in range(n_decoy):
        droot = next(it)
        ops.append(Operation(kind="premise", content=droot))
        cur = droot
        drules: List[str] = []
        dlits: List[str] = []
        for j in range(depth):
            ridx[0] += 1
            nm = f"r_{ridx[0]}"
            nxt = claim if j == depth - 1 else next(it)
            ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
            drules.append(nm)
            dlits.append(nxt)
            cur = nxt
        # rotate by level as well as by index: n_decoy caps at 3, so `k % 4` alone would
        # never reach the fourth kind
        mode = (k + level) % 4
        if mode == 3:
            # DEAD AT AN IMPOSSIBILITY AXIOM. The route reaches the claim and carries no visible attack
            # on any of its rules -- it is dead because its root asserts something an axiom declares
            # impossible. Verified: the root is OVERRULED and the whole route with it, while a model
            # tracing rules sees a complete, unattacked chain.
            ops.append(Operation(kind="axiom", content="-" + droot))
            where = "impossible-root"
        elif mode == 0:
            ops.append(Operation(kind="premise", content="-" + droot))
            ops.append(Operation(kind="prefer_premise", stronger="-" + droot, weaker=droot))
            where = "root"
        elif mode == 1:
            _tower(ops, it, ridx, dlits[len(dlits) // 2], 1)
            where = "mid"
        else:
            src = next(it)
            ops.append(Operation(kind="premise", content=src))
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"w_{ridx[0]}", antecedents=(src,),
                                 consequent="-" + drules[-1]))
            where = "near-claim"
        decoy_info.append({"rules": drules, "defeat_at": where})

    if branch_decoys and depth >= 4:
        anchor = line_ops[1 + depth // 3].consequent
        cur = anchor
        for j in range(2):
            ridx[0] += 1
            nxt = next(it)
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(cur,),
                                 consequent=nxt))
            cur = nxt

    for _ in range(n_filler):
        a, b = next(it), next(it)
        ops.append(Operation(kind="premise", content=a))
        ridx[0] += 1
        ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(a,),
                             consequent=b))

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    ops = list(ops) + _lx
    ops, _map = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    line_ops = [o for o in ops
                if (o.kind == "premise" and o.content == root)
                or (o.kind in ("defeasible", "strict") and o.name in
                    {_map.get(x.name, x.name) for x in line_ops if getattr(x, "name", None)})]
    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))

    atoms, rnames = split_atoms_and_rules(base)
    if atoms & rnames:
        return None
    if status(base, claim, ordering) != "JUSTIFIED":
        return None

    for d in decoy_info:
        sub = [o for o in base if not (o.kind == "defeasible" and o.name in d["rules"])]
        if status(sub, claim, ordering) != "JUSTIFIED":
            return None
    without_true = [o for o in base
                    if not (o.kind == "defeasible" and o.name == line_ops[1].name)]
    if status(without_true, claim, ordering) == "JUSTIFIED":
        return None

    ref_lines = [render_op(o) for o in line_ops]
    prompt = _render_prompt(render_ops(base), claim, ordering)
    return CCItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, claim=claim,
        line_ops=line_ops, ordering=ordering, level=level,
        reference="[answer]\n" + "\n".join(ref_lines) + "\n[/answer]",
        metadata={
            "chain_depth": depth, "n_decoys": n_decoy, "tower_height_true": tower_true,
            "decoy_defeat_points": [d["defeat_at"] for d in decoy_info],
            "branch_decoys": branch_decoys,
            "n_items": len(base), "line_length": len(line_ops),
            "negated_line_root": use_neg_root,
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
        })


def _render_prompt(theory: str, claim: str, ordering: str) -> str:
    on = "the last-link strength ordering" if ordering == LAST_LINK \
        else "the weakest-link strength ordering"
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"The claim {claim} is justified.\n"
            f"Write all and only the directives that form the argumentation line justifying {claim}, "
            "in order from the premise to the claim.\n\n"
            "Answer format: one directive per line, copied exactly as it appears above, between "
            "[answer] and [/answer].")


# Answer region comes from core.scoring: the LAST complete region, so a
# reasoning model that drafts and then revises is scored on the revision.
from core.scoring import answer_region


def score(answer_text: str, item: CCItem) -> Dict:
    diag: Dict = {"n_quoted": 0, "n_gold": len(item.line_ops), "extra": [], "missing": []}
    m = answer_region(answer_text)
    if m is None:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    body = m
    quoted = re.findall(r"\[[^\]]*\]", body)
    if not quoted:
        quoted = [l.strip() for l in body.splitlines() if l.strip()]
    else:
        residue = re.sub(r"\[[^\]]*\]", " ", body)
        junk = [t for t in residue.split()
                if t.strip(",;.-*\u2022()") and not re.fullmatch(r"\d+[.)]?", t)]
        if junk:
            diag["n_unparseable"] = len(junk)
            diag["junk_tokens"] = junk[:6]
            return {"score": 0.0, "reason": f"unparseable_tokens:{len(junk)}",
                    "diagnostics": diag}
    diag["n_quoted"] = len(quoted)
    if not quoted:
        return {"score": 0.0, "reason": "empty_answer", "diagnostics": diag}

    gold_lines = [render_op(o) for o in item.line_ops]
    gold_set, pred_set = set(gold_lines), set(quoted)
    tp = len(gold_set & pred_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gold_set), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    diag["extra"] = sorted(pred_set - gold_set)[:5]
    diag["missing"] = sorted(gold_set - pred_set)[:5]

    by_line = {render_op(o): o for o in item.base_ops}
    picked = [by_line[l] for l in quoted if l in by_line]
    justifies = status(picked, item.claim, item.ordering) == "JUSTIFIED" if picked else False

    return {"score": round(f1, 4), "reason": "ok",
            "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "exact_match": pred_set == gold_set,
            "correct_order": quoted == gold_lines,
            "behaviourally_justifies": justifies,
            "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              tries: int = 14) -> Optional[CCItem]:
    for k in range(tries):
        it = build(level, seed * 71 + k, ordering)
        if it is not None:
            return it
    return None
