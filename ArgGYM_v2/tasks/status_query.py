from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier
from core.invariants import randomize_rule_names, split_atoms_and_rules
from core.curriculum import junction_budget, JUNCTION_CAPS, junctions_for, PROFILES

TASK = "status_query"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
EASY_LEVELS = 4
MAX_STATUS_SHARE = 0.45
PANEL_THRESHOLD = round(MAX_STATUS_SHARE + 0.03, 3)

ATTACK_HEURISTIC_EXPECTED = 0.56
_L = "abcdefghijklmnopqrstuvwxy"
STATUSES = ("JUSTIFIED", "OVERRULED", "UNDECIDED")


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


def full_status_map(ops: Sequence[Operation], ordering: str) -> Dict[str, str]:
    try:
        v = ASPICVerifier.from_operations(list(ops), ordering=ordering)
        return {k: str(x) for k, x in v.status_map().items()}
    except Exception:
        return {}


@dataclass
class SQItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    queried: List[str]
    gold: Dict[str, str]
    ordering: str
    level: int
    reference: str
    metadata: Dict = field(default_factory=dict)


def _tower(ops: List[Operation], names, ridx: List[int], target_lit: str, height: int) -> None:
    prev = None
    for i in range(height):
        root = next(names)
        ops.append(Operation(kind="premise", content=root))
        ridx[0] += 1
        nm = f"w_{ridx[0]}"
        cons = ("-" + target_lit) if i == 0 else ("-" + prev)
        ops.append(Operation(kind="defeasible", name=nm, antecedents=(root,), consequent=cons))
        prev = nm


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Optional[SQItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "sq", profile))
    prof = PROFILES[profile]
    n_query = max(3, round(3 + (level - 1) * (40 - 3) / 14))
    n_group = max(3, round(n_query / 1.6))
    max_tower = 0 if level < 5 else min(1 + (level - 5) // 4, 3)
    # JUNCTIONS from level 8. Cutting either branch kills the conclusion, so a claim above a junction
    # can be overruled by a defeat on a branch that never mentions it.
    # Sized from the EXPECTED rule count so the share is constant across the curriculum.
    # A level-scaled budget gave 3% at level 5 and 21% at level 15 from the same machinery.
    j_budget = junctions_for(level, max(1, n_group * 3))
    j_used = [0]

    names = _names(stable_seed(seed, (level, ordering, "nm") * 3), 40 + n_group * 12)
    it = iter(names)
    ops: List[Operation] = []
    ridx = [0]
    planned: List[Tuple[str, str]] = []

    for g in range(n_group):
        want = STATUSES[g % 3]
        root, mid = next(it), next(it)

        if j_used[0] < j_budget and prof.permits("defeasible"):
            # a junction whose SECOND branch is dead, so the conclusion is overruled even though
            # nothing attacks it and its first branch is healthy
            b1, b2, jt = next(it), next(it), next(it)
            ops.append(Operation(kind="premise", content=b1))
            ops.append(Operation(kind="premise", content=b2))
            ops.append(Operation(kind="premise", content="-" + b2))
            ops.append(Operation(kind="prefer_premise", stronger="-" + b2, weaker=b2))
            ridx[0] += 1
            s1 = f"r_{ridx[0]}"
            l1 = next(it)
            ops.append(Operation(kind="defeasible", name=s1, antecedents=(b1,), consequent=l1))
            ridx[0] += 1
            s2 = f"r_{ridx[0]}"
            l2 = next(it)
            ops.append(Operation(kind="defeasible", name=s2, antecedents=(b2,), consequent=l2))
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                 antecedents=(l1, l2), consequent=jt))
            planned.append((jt, "OVERRULED"))
            planned.append((l1, "JUSTIFIED"))
            j_used[0] += 1
            continue

        if not prof.permits("defeasible"):
            # P_S: no defeasible rules, so neither undercut nor rebut is available. The only lever is
            # a premise preference on the root, which is why this fragment is a different problem
            # rather than an easier one.
            ops.append(Operation(kind="premise", content=root))
            ridx[0] += 1
            ops.append(Operation(kind="strict", name=f"r_{ridx[0]}",
                                 antecedents=(root,), consequent=mid))
            if want != "JUSTIFIED":
                ops.append(Operation(kind="premise", content="-" + root))
                if want == "OVERRULED":
                    ops.append(Operation(kind="prefer_premise",
                                         stronger="-" + root, weaker=root))
            planned.append((mid, want))
            continue

        if want == "JUSTIFIED":
            ops.append(Operation(kind="premise", content=root))
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(root,),
                                 consequent=mid))
            if max_tower and rng.random() < 0.75:
                _tower(ops, it, ridx, mid, 2 * rng.randint(1, max_tower))
        elif want == "UNDECIDED":
            upstream = rng.random() < 0.5
            ops.append(Operation(kind="premise", content=root))
            h = (2 * rng.randint(0, max_tower) + 1) if max_tower else 1
            if upstream:
                stem = next(it)
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(root,),
                                     consequent=stem))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(stem,),
                                     consequent=mid))
                _tower(ops, it, ridx, stem, h)
            else:
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(root,),
                                     consequent=mid))
                _tower(ops, it, ridx, mid, h)
        else:
            far = rng.random() < 0.5
            if far:
                stem = next(it)
                ops.append(Operation(kind="premise", content=root))
                ops.append(Operation(kind="premise", content="-" + root))
                ops.append(Operation(kind="prefer_premise", stronger="-" + root, weaker=root))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(root,),
                                     consequent=stem))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(stem,),
                                     consequent=mid))
            else:
                ops.append(Operation(kind="premise", content=root))
                ridx[0] += 1
                sup = f"r_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=sup, antecedents=(root,),
                                     consequent=mid))
                q = next(it)
                ops.append(Operation(kind="premise", content=q))
                ridx[0] += 1
                atk = f"w_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=atk, antecedents=(q,),
                                     consequent="-" + mid))
                ops.append(Operation(kind="prefer_rule", stronger=atk, weaker=sup))
        planned.append((mid, want))

        # The axiom root and the strict step are gated SEPARATELY. Tying them together meant P_S_D,
        # which forbids axioms but permits strict rules, produced no strict rules at all and was
        # indistinguishable from P_D.
        if g % 4 == 1 and prof.permits("strict"):
            a2, b2 = next(it), next(it)
            ops.append(Operation(kind="axiom" if prof.permits("axiom") else "premise",
                                 content=a2))
            ridx[0] += 1
            ops.append(Operation(kind="strict", name=f"r_{ridx[0]}", antecedents=(a2,),
                                 consequent=b2))
            if max_tower and rng.random() < 0.7:
                _tower(ops, it, ridx, b2, 2 * rng.randint(1, max_tower))
            planned.append((b2, "JUSTIFIED"))
        if g % 5 == 2 and prof.permits("prefer_premise"):
            c1, c2 = next(it), next(it)
            ops.append(Operation(kind="premise", content=c1))
            ops.append(Operation(kind="premise", content="-" + c1))
            ops.append(Operation(kind="prefer_premise", stronger="-" + c1, weaker=c1))
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=("-" + c1,),
                                 consequent=c2))
            if max_tower and rng.random() < 0.7:
                _tower(ops, it, ridx, c2, 2 * rng.randint(1, max_tower))
            planned.append((c2, "JUSTIFIED"))

    ops = prof.filter(ops)
    ops, _rmap = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))
    atoms, rnames = split_atoms_and_rules(base)
    if atoms & rnames:
        return None

    sm = full_status_map(base, ordering)
    if not sm:
        return None

    by_status: Dict[str, List[str]] = collections.defaultdict(list)
    for lit, stat in sm.items():
        if lit.startswith("-"):
            continue
        if stat in STATUSES:
            by_status[stat].append(lit)
    for k in by_status:
        by_status[k].sort()
        rng.shuffle(by_status[k])
    if any(len(by_status.get(s, [])) == 0 for s in STATUSES):
        return None

    per = max(1, n_query // 3)
    queried: List[str] = []
    for s in STATUSES:
        queried.extend(by_status[s][:per])
    pool = [l for s in STATUSES for l in by_status[s][per:]]
    rng.shuffle(pool)
    for l in pool:
        if len(queried) >= n_query:
            break
        cand = queried + [l]
        c = collections.Counter(sm[x] for x in cand)
        if max(c.values()) / len(cand) <= MAX_STATUS_SHARE:
            queried.append(l)
    rng.shuffle(queried)

    gold = {l: sm[l] for l in queried}
    counts = collections.Counter(gold.values())
    if len(counts) < 3:
        return None
    if max(counts.values()) / len(gold) > MAX_STATUS_SHARE:
        return None

    lines = [f"{l}: {gold[l].lower()}" for l in queried]
    prompt = _render_prompt(render_ops(base), queried, ordering)
    return SQItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, queried=queried, gold=gold,
        ordering=ordering, level=level,
        reference="[answer]\n" + "\n".join(lines) + "\n[/answer]",
        metadata={
            "n_queried": len(queried), "n_groups": n_group, "max_tower": max_tower,
            "n_items": len(base), "target_n_query": n_query,
            "status_counts": dict(counts),
            "modal_share": round(max(counts.values()) / len(gold), 4),
            "profile": profile,
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
            "n_axioms": len([o for o in base if o.kind == "axiom"]),
            "n_strict": len([o for o in base if o.kind == "strict"]),
        })


def _render_prompt(theory: str, queried: Sequence[str], ordering: str) -> str:
    on = "the last-link strength ordering" if ordering == LAST_LINK \
        else "the weakest-link strength ordering"
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"State the status of each of the following claims: {', '.join(queried)}.\n"
            "Possible statuses: justified, overruled, undecided.\n\n"
            "Answer format: one line per claim, written as `claim: status`, "
            "between [answer] and [/answer].")


_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)
_PAIR = re.compile(r"(-?\w+)\s*[:=]\s*(justified|overruled|undecided)\b(?!\w)", re.I)


def score(answer_text: str, item: SQItem) -> Dict:
    diag: Dict = {"n_predicted": 0, "n_gold": len(item.gold), "wrong": [], "missing": []}
    m = _ANSWER.search(answer_text or "")
    if m is None:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    body = m.group(1)
    pred: Dict[str, str] = {}
    for claim, stat in _PAIR.findall(body):
        pred[claim] = stat.upper()
    residue = _PAIR.sub(" ", body)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    diag["n_unparseable"] = len(junk)
    diag["junk_tokens"] = junk[:6]
    if junk:
        return {"score": 0.0, "reason": f"unparseable_tokens:{len(junk)}",
                "diagnostics": diag}
    diag["n_predicted"] = len(pred)
    if not pred:
        return {"score": 0.0, "reason": "no_parseable_lines", "diagnostics": diag}

    gold_pairs = set(item.gold.items())
    pred_pairs = set(pred.items())
    tp = len(gold_pairs & pred_pairs)
    precision = tp / max(len(pred_pairs), 1)
    recall = tp / max(len(gold_pairs), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    diag["wrong"] = sorted(f"{k}:said {v}, is {item.gold[k]}"
                           for k, v in pred.items() if k in item.gold and item.gold[k] != v)[:6]
    diag["missing"] = sorted(k for k in item.gold if k not in pred)[:6]
    diag["n_correct"] = tp
    return {"score": round(f1, 4), "reason": "ok",
            "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "exact_match": pred_pairs == gold_pairs, "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              profile: str = "FULL", tries: int = 24) -> Optional[SQItem]:
    for k in range(tries):
        it = build(level, seed * 97 + k, ordering, profile)
        if it is not None:
            return it
    return None
