from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier
from core.curriculum import (junction_budget, JUNCTION_CAPS, PROFILES, wants_ternary,
                            junctions_for, negated_branch)
from core.invariants import (split_atoms_and_rules, randomize_rule_names, negation_gadget,
                        language_enrichment)

TASK = "claim_chain"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
EASY_LEVELS = 3
_L = "abcdefghijklmnopqrstuvwxy"


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def _names(seed: int, n: int) -> List[str]:
    """A shuffled pool of distinct three-character atoms.

    Callers consume this through a plain iterator, so a pool shorter than the demand fails deep inside
    construction rather than at the point of allocation. Junctions raised demand -- each adds a branch
    root and a branch literal -- and the shortfall surfaced as a rule with a None antecedent. The
    request is asserted rather than silently truncated."""
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:14] for b in _L[10:] for d in range(10)]
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"name pool exhausted: asked {n}, have {len(pool)}")
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


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Optional[CCItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "cc"))
    depth = max(2, min(2 + level, 20))
    n_decoy = 1 if level < 4 else min(1 + (level - 4) // 4, 3)
    tower_true = 0 if level < 8 else 2 * min(1 + (level - 8) // 4, 3)
    n_filler = max(0, min(4 + level * 3, 60))
    branch_decoys = level >= 11
    # JUNCTIONS from level 8. A rule with two antecedents is not decoration: cutting EITHER branch
    # kills the conclusion, so the traced line must include both branches, and under weakest-link the
    # argument is only as strong as its weaker branch. Every rule in the suite was previously linear --
    # 0 of 533 had more than one antecedent -- so no task exercised this at all.
    # Base junctions on the TRACED LINE stay modest -- each adds two or three directives to
    # the answer. The filler top-up below carries the theory to the target share, which is
    # theory-only capacity and costs nothing in the answer.
    j_budget = min(3, junction_budget(level, JUNCTION_CAPS["claim_chain"]))

    # Pool sized for junctions: each consumes a branch root and a branch literal, and a
    # ternary junction consumes two of each. A blind multiplier was applied to the SEED
    # argument by mistake, which left the size unchanged and produced StopIteration deep
    # in construction.
    names = _names(stable_seed(seed, level, ordering, "nm"),
                   ((60 + depth * 3 + n_decoy * (depth + 6) * 4) + 6 * j_budget) + n_filler * 2 + tower_true * 3)
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
    # Junction points spread along the traced line, one per budgeted junction.
    j_points = set()
    if j_budget and depth >= 3:
        step = max(1, depth // (j_budget + 1))
        j_points = {min(depth - 2, step * (i + 1)) for i in range(j_budget)}
    for j in range(depth):
        ridx[0] += 1
        nm = f"r_{ridx[0]}"
        nxt = claim if j == depth - 1 else next(it)
        if j in j_points:
            # EXTRA BRANCHES feeding this step. Cutting any branch kills everything above it, and the
            # traced line must contain them all, so the answer is no longer a simple path. From level
            # 9 a share of junctions take a THIRD antecedent, which tests tracking several
            # simultaneous dependencies rather than just one alternative.
            n_extra = 2 if wants_ternary(level, sorted(j_points).index(j)) else 1
            extra_lits = []
            for _e in range(n_extra):
                # a share of branches fire from a NEGATED root, which is the negated_antecedent role
                _neg = negated_branch(sorted(j_points).index(j) * 2 + _e)
                broot = next(it)
                _src = ("-" + broot) if _neg else broot
                ops.append(Operation(kind="premise", content=_src))
                ridx[0] += 1
                bname = f"r_{ridx[0]}"
                blit = next(it)
                brule = Operation(kind="defeasible", name=bname, antecedents=(_src,),
                                  consequent=blit)
                ops.append(brule)
                extra_lits.append(blit)
                line_ops.append(Operation(kind="premise", content=_src))
                line_ops.append(brule)
            ridx[0] += 1
            nm = f"r_{ridx[0]}"
            r = Operation(kind="defeasible", name=nm,
                          antecedents=tuple([cur] + extra_lits), consequent=nxt)
            ops.append(r)
            line_ops.append(r)
        else:
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
        # DECOY chains carry junctions as well as the traced line. Confining junctions to the true
        # line meant every one added two or three directives to the ANSWER, so density could not pass
        # about 7% without the answer ballooning. Decoys and filler are theory-only capacity.
        _dj = {max(1, depth // 2)} if (level >= 5 and depth >= 3) else set()
        if level >= 9 and depth >= 5:
            _dj.add(max(1, depth // 4))
        for j in range(depth):
            ridx[0] += 1
            nm = f"r_{ridx[0]}"
            nxt = claim if j == depth - 1 else next(it)
            if j in _dj:
                _ex = []
                for _e in range(2 if wants_ternary(level, k) else 1):
                    br, bl = next(it), next(it)
                    ops.append(Operation(kind="premise", content=br))
                    ridx[0] += 1
                    ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                         antecedents=(br,), consequent=bl))
                    _ex.append(bl)
                ridx[0] += 1
                nm = f"r_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=nm,
                                     antecedents=tuple([cur] + _ex), consequent=nxt))
            else:
                ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,),
                                     consequent=nxt))
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
            ops.append(Operation(
                kind="axiom" if PROFILES[profile].permits("axiom") else "premise",
                content="-" + droot))
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
        # Anchor on a RULE of the traced line, chosen by filtering rather than by index.
        #
        # This was `line_ops[1 + depth // 3].consequent`, which assumed line_ops alternated
        # premise-then-rules. A junction adds THREE entries -- a branch premise, a branch rule and the
        # junction rule -- so with more than one junction the index landed on a premise and took its
        # `consequent`, which is None. That produced a rule with a None antecedent and an
        # AttributeError deep inside split_atoms_and_rules, several frames from the cause.
        #
        # This is why claim_chain was capped at one junction: the cap hid the bug rather than fixing
        # it, and the note recording that said the cause "was not isolated".
        _line_rules = [o for o in line_ops if o.kind in ("defeasible", "strict") and o.consequent]
        if not _line_rules:
            return None
        anchor = _line_rules[min(len(_line_rules) - 1, depth // 3)].consequent
        cur = anchor
        for j in range(2):
            ridx[0] += 1
            nxt = next(it)
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(cur,),
                                 consequent=nxt))
            cur = nxt

    # FILLER carries junctions at the target share. Filler exists to pad the theory, so making a
    # fifth of it multi-antecedent costs nothing and is the cheapest density available.
    # Size the top-up against the FINAL rule count, not the count so far. Each junction adds its own
    # branch rules, so a budget computed before they exist undershoots -- measured 13% where 20% was
    # asked. Every junction placed here contributes one multi-antecedent rule and `1 + extras` single
    # ones, so the fixed point of `need = share * (base + placed * (2 + extras))` is what to solve.
    _base = sum(1 for o in ops if o.kind == "defeasible")
    _have = sum(1 for o in ops if o.kind == "defeasible" and len(o.antecedents or ()) > 1)
    _fill_j = 0
    for _try in range(0, n_filler + 1):
        _total = _base + n_filler + _try * 2          # each junction adds ~2 extra branch rules
        if (_have + _try) >= junctions_for(level, _total, solve=False):
            _fill_j = _try
            break
    else:
        _fill_j = n_filler
    for _fi in range(n_filler):
        a, b = next(it), next(it)
        ops.append(Operation(kind="premise", content=a))
        ridx[0] += 1
        if _fi < _fill_j:
            _ex = []
            for _e in range(2 if (level >= 9 and _fi % 2 == 0) else 1):
                br, bl = next(it), next(it)
                _src = ("-" + br) if negated_branch(_fi * 2 + _e) else br
                ops.append(Operation(kind="premise", content=_src))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                     antecedents=(_src,), consequent=bl))
                _ex.append(bl)
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                 antecedents=tuple([a] + _ex), consequent=b))
        else:
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(a,),
                                 consequent=b))

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    ops = list(ops) + _lx
    ops, _map = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    # Rebuild the line after renaming. It must keep EVERY premise the line depends on, not just the
    # main root: a junction step draws on a second branch with its own root, and dropping it leaves a
    # reference whose rules cannot fire. The behavioural gate caught exactly that.
    _line_roots = {x.content for x in line_ops if x.kind == "premise"}
    _line_rules = {_map.get(x.name, x.name) for x in line_ops if getattr(x, "name", None)}
    line_ops = [o for o in ops
                if (o.kind in ("premise", "axiom") and o.content in _line_roots)
                or (o.kind in ("defeasible", "strict") and o.name in _line_rules)]
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


_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)


def score(answer_text: str, item: CCItem) -> Dict:
    diag: Dict = {"n_quoted": 0, "n_gold": len(item.line_ops), "extra": [], "missing": []}
    m = _ANSWER.search(answer_text or "")
    if m is None:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    body = m.group(1)
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

    # BEHAVIOURAL CHECK, evaluated IN THE THEORY rather than in isolation.
    #
    # The earlier version ran the quoted directives alone through the engine. That is the wrong
    # question and it produced a false positive: a DECOY line quoted by itself has no defeater present,
    # so it stands, and `behaviourally_justifies` reported True on an answer scoring F1 = 0.000.
    # Verified that no genuine second answer exists -- every decoy IS defeated in the theory as given,
    # 0 undefeated decoy routes over 6 items -- so the diagnostic was mislabelling, not the generator
    # admitting an alternative.
    #
    # The right question is whether the quoted set is a line that carries the claim IN THE THEORY: it
    # must justify the claim on its own AND every rule in it must still be undefeated once the rest of
    # the theory is present.
    by_line = {render_op(o): o for o in item.base_ops}
    picked = [by_line[l] for l in quoted if l in by_line]
    justifies = False
    if picked:
        alone = status(picked, item.claim, item.ordering) == "JUSTIFIED"
        if alone:
            try:
                v = ASPICVerifier.from_operations(list(item.base_ops), ordering=item.ordering)
                # in the full theory every quoted literal must still hold, and the claim with them
                lits = [o.consequent for o in picked
                        if o.kind in ("defeasible", "strict") and o.consequent]
                justifies = (str(v.status(item.claim)) == "JUSTIFIED"
                             and all(str(v.status(x)) == "JUSTIFIED" for x in lits))
            except Exception:
                justifies = False
    diag["behavioural_in_theory"] = justifies

    return {"score": round(f1, 4), "reason": "ok",
            "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "exact_match": pred_set == gold_set,
            "correct_order": quoted == gold_lines,
            "behaviourally_justifies": justifies,
            "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK, profile: str = "FULL",
              tries: int = 14) -> Optional[CCItem]:
    for k in range(tries):
        it = build(level, seed * 71 + k, ordering, profile)
        if it is not None:
            return it
    return None
