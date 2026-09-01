from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation, UNSATISFIABLE
from arggym.aspic.api import ASPICVerifier
from arggym.core.curriculum import junction_budget, JUNCTION_CAPS
from arggym.core.invariants import randomize_rule_names, split_atoms_and_rules

TASK = "semantics_query"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
EASY_LEVELS = 3
MAX_DIRECTIVES = 26

MAX_EAGER_ARGUMENTS = 18
MAX_STATUS_SHARE = 0.45
PANEL_THRESHOLD = round(MAX_STATUS_SHARE + 0.03, 3)
_L = "abcdefghijklmnopqrstuvwxy"

GROUNDED = "grounded"
SCEPT_PREF = "sceptical preferred"
CRED_PREF = "credulous preferred"
STABLE = "stable"
EAGER = "eager"

SEMANTICS_BY_LEVEL = {
    1: (GROUNDED,),
    4: (GROUNDED, SCEPT_PREF),
    8: (GROUNDED, SCEPT_PREF, CRED_PREF),
    12: (GROUNDED, SCEPT_PREF, CRED_PREF, STABLE),
    14: (GROUNDED, SCEPT_PREF, CRED_PREF, STABLE, EAGER),
}


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def _names(seed: int, n: int) -> List[str]:
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:14] for b in _L[10:] for d in range(10)]
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"name pool exhausted: asked {n}")
    return pool[:n]


def semantics_for(level: int) -> Tuple[str, ...]:
    out = SEMANTICS_BY_LEVEL[1]
    for k in sorted(SEMANTICS_BY_LEVEL):
        if level >= k:
            out = SEMANTICS_BY_LEVEL[k]
    return out


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


@dataclass
class SemCache:
    """Each semantics a theory was asked for, enumerated once."""
    grounded: Optional[Dict[str, str]] = None
    preferred: Optional[List[set]] = None
    stable: Optional[List[set]] = None
    eager: Optional[set] = None


def semantics_cache(ops: Sequence[Operation], sems: Sequence[str], ordering: str,
                    verifier: Optional[ASPICVerifier] = None) -> Optional[SemCache]:
    """Enumerate every requested semantics once, over the one theory the queries share.

    Returns None on the failures that used to reject an item query by query: the
    verifier not building, or an enumeration raising.
    """
    try:
        v = verifier if verifier is not None else ASPICVerifier.from_operations(
            list(ops), ordering=ordering)
    except Exception:
        return None
    cache = SemCache()
    try:
        if GROUNDED in sems:
            cache.grounded = v.status_map()
        if SCEPT_PREF in sems or CRED_PREF in sems:
            cache.preferred = v.preferred_conclusions()
        if STABLE in sems:
            cache.stable = v.stable_conclusions()
        if EAGER in sems:
            cache.eager = {str(getattr(x, "conclusion", x)) for x in v.fw.eager_extension()}
    except Exception:
        return None
    return cache


def status_under(cache: SemCache, claim: str, semantics: str) -> Optional[str]:
    if semantics == GROUNDED:
        return cache.grounded.get(claim.strip(), UNSATISFIABLE)
    if semantics in (SCEPT_PREF, CRED_PREF):
        exts = cache.preferred
        if not exts:
            return None
        if semantics == SCEPT_PREF:
            if all(claim in e for e in exts):
                return "JUSTIFIED"
            if any(claim in e for e in exts):
                return "UNDECIDED"
            return "OVERRULED"
        return "JUSTIFIED" if any(claim in e for e in exts) else "OVERRULED"
    if semantics == EAGER:
        eager = cache.eager
        if claim in eager:
            return "JUSTIFIED"
        return "OVERRULED" if ("-" + claim if not claim.startswith("-")
                               else claim[1:]) in eager else "UNDECIDED"
    if semantics == STABLE:
        exts = cache.stable
        if not exts:
            return "NO_STABLE_EXTENSION"
        if all(claim in e for e in exts):
            return "JUSTIFIED"
        if any(claim in e for e in exts):
            return "UNDECIDED"
        return "OVERRULED"
    return None


@dataclass
class SemItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    queries: List[Tuple[str, str]]
    gold: Dict[Tuple[str, str], str]
    ordering: str
    level: int
    reference: str
    metadata: Dict = field(default_factory=dict)


def _floating(it, ridx) -> Tuple[List[Operation], str, str]:
    a, b, p, f = next(it), next(it), next(it), next(it)
    ridx[0] += 1
    r1 = f"r_{ridx[0]}"
    ridx[0] += 1
    r2 = f"r_{ridx[0]}"
    ridx[0] += 1
    r3 = f"r_{ridx[0]}"
    ridx[0] += 1
    r4 = f"r_{ridx[0]}"
    ops = [Operation(kind="premise", content=a), Operation(kind="premise", content=b),
           Operation(kind="defeasible", name=r1, antecedents=(a,), consequent=p),
           Operation(kind="defeasible", name=r2, antecedents=(b,), consequent="-" + p),
           Operation(kind="defeasible", name=r3, antecedents=(p,), consequent=f),
           Operation(kind="defeasible", name=r4, antecedents=("-" + p,), consequent=f)]
    return ops, f, p


def _settled(it, ridx) -> Tuple[List[Operation], str]:
    a, b, c, x = next(it), next(it), next(it), next(it)
    ridx[0] += 1
    r1 = f"r_{ridx[0]}"
    ridx[0] += 1
    r2 = f"r_{ridx[0]}"
    ridx[0] += 1
    r3 = f"r_{ridx[0]}"
    ops = [Operation(kind="premise", content=a), Operation(kind="premise", content=b),
           Operation(kind="premise", content=c),
           Operation(kind="defeasible", name=r1, antecedents=(a,), consequent=x),
           Operation(kind="defeasible", name=r2, antecedents=(b,), consequent="-" + r1),
           Operation(kind="defeasible", name=r3, antecedents=(c,), consequent="-" + r2)]
    return ops, x


def _self_undermining(it, ridx):
    from arggym.aspic.engine import Operation
    a, p, q = next(it), next(it), next(it)
    names = []
    for _ in range(3):
        ridx[0] += 1
        names.append(f"r_{ridx[0]}")
    ops = [Operation(kind="premise", content=a),
           Operation(kind="defeasible", name=names[0], antecedents=(a,), consequent="-" + p),
           Operation(kind="defeasible", name=names[1], antecedents=("-" + p,), consequent=q),
           Operation(kind="defeasible", name=names[2], antecedents=(q,), consequent=p)]
    return ops, q


def _junction_cluster(it, ridx, ternary: bool):
    from arggym.aspic.engine import Operation
    roots = [next(it) for _ in range(3 if ternary else 2)]
    lits = [next(it) for _ in roots]
    concl = next(it)
    ops = []
    for r, l in zip(roots, lits):
        ops.append(Operation(kind="premise", content=r))
        ridx[0] += 1
        ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(r,),
                             consequent=l))
    ridx[0] += 1
    ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                         antecedents=tuple(lits), consequent=concl))
    ops.append(Operation(kind="premise", content="-" + roots[0]))
    return ops, concl


def _defeated(it, ridx):
    from arggym.aspic.engine import Operation
    a, b, x = next(it), next(it), next(it)
    ridx[0] += 1
    pro = f"r_{ridx[0]}"
    ridx[0] += 1
    con = f"r_{ridx[0]}"
    ops = [Operation(kind="premise", content=a), Operation(kind="premise", content=b),
           Operation(kind="defeasible", name=pro, antecedents=(a,), consequent=x),
           Operation(kind="defeasible", name=con, antecedents=(b,), consequent="-" + x),
           Operation(kind="prefer_rule", stronger=con, weaker=pro),
           Operation(kind="prefer_premise", stronger=b, weaker=a)]
    return ops, x


def _odd(it, ridx) -> Tuple[List[Operation], str]:
    a, b, c, x, y, z = (next(it) for _ in range(6))
    names = []
    for _ in range(6):
        ridx[0] += 1
        names.append(f"r_{ridx[0]}")
    ops = [Operation(kind="premise", content=a), Operation(kind="premise", content=b),
           Operation(kind="premise", content=c),
           Operation(kind="defeasible", name=names[0], antecedents=(a,), consequent=x),
           Operation(kind="defeasible", name=names[1], antecedents=(b,), consequent="-" + x),
           Operation(kind="defeasible", name=names[2], antecedents=(b,), consequent=y),
           Operation(kind="defeasible", name=names[3], antecedents=(c,), consequent="-" + y),
           Operation(kind="defeasible", name=names[4], antecedents=(c,), consequent=z),
           Operation(kind="defeasible", name=names[5], antecedents=(a,), consequent="-" + z)]
    return ops, x


def build(level: int, seed: int, ordering: str = LAST_LINK) -> Optional[SemItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "sem"))
    sems = semantics_for(level)
    n_cluster = 2

    it = iter(_names(stable_seed(seed, level, ordering, "nm"), 40 + n_cluster * 12))
    ops: List[Operation] = []
    ridx = [0]
    candidates: List[str] = []

    kinds = ["defeated", "settled", "odd"]
    j_budget = junction_budget(level, JUNCTION_CAPS.get("semantics_query", 3))
    _n_cluster = 2 + (1 if (j_budget or EAGER in sems) else 0)
    for k in range(_n_cluster):
        kind = ("floating" if k == 0 else "defeated" if k == 1
                else "undermining" if EAGER in sems else "junction")
        if kind == "floating":
            block, shared, contested = _floating(it, ridx)
            candidates.extend([shared, contested])
        elif kind == "undermining":
            block, x = _self_undermining(it, ridx)
            candidates.append(x)
        elif kind == "junction":
            block, x = _junction_cluster(it, ridx, ternary=False)
            candidates.append(x)
        elif kind == "defeated":
            block, x = _defeated(it, ridx)
            candidates.append(x)
        elif kind == "settled":
            block, x = _settled(it, ridx)
            candidates.append(x)
        else:
            block, x = _odd(it, ridx)
            candidates.append(x)
        ops.extend(block)

    ops, _map = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    atoms, rnames = split_atoms_and_rules(ops)
    if atoms & rnames:
        return None
    if len(ops) > MAX_DIRECTIVES:
        return None

    rng.shuffle(ops)
    facts = [o for o in ops if o.kind in ("premise", "axiom")]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    prefs = [o for o in ops if o.kind.startswith("prefer")]
    base = facts + rules + prefs

    _cache_v: Optional[ASPICVerifier] = None
    if EAGER in sems:
        try:
            _v = ASPICVerifier.from_operations(list(base), ordering=ordering)
            _na = len(_v.fw.af.arguments)
        except Exception:
            return None
        _fill = 0
        while _na + 2 <= MAX_EAGER_ARGUMENTS and _fill < 8:
            _a, _c = next(it), next(it)
            ridx[0] += 1
            base.append(Operation(kind="premise", content=_a))
            base.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                  antecedents=(_a,), consequent=_c))
            candidates.append(_c)
            _na += 2
            _fill += 1
        try:
            _v = ASPICVerifier.from_operations(list(base), ordering=ordering)
            if len(_v.fw.af.arguments) > MAX_EAGER_ARGUMENTS:
                return None
        except Exception:
            return None
        _cache_v = _v

    cache = semantics_cache(base, sems, ordering, verifier=_cache_v)
    if cache is None:
        return None
    gold: Dict[Tuple[str, str], str] = {}
    for c in candidates:
        for s in sems:
            st = status_under(cache, c, s)
            if st is None:
                return None
            gold[(c, s)] = st
    if not gold:
        return None

    by_claim: Dict[str, List[str]] = {}
    for (c, s) in gold:
        by_claim.setdefault(c, []).append(s)
    diverging = [c for c in by_claim if len({gold[(c, s)] for s in by_claim[c]}) > 1]
    if level > EASY_LEVELS and not diverging:
        return None

    by_status = collections.defaultdict(list)
    for k, v in gold.items():
        by_status[v].append(k)
    for v in by_status:
        by_status[v].sort()
        rng.shuffle(by_status[v])
    order = sorted(by_status, key=lambda v: -len(by_status[v]))
    queries = []
    while any(by_status[v] for v in order):
        for v in order:
            if by_status[v]:
                queries.append(by_status[v].pop())
    _first_of = {}
    for q in queries:
        _first_of.setdefault(q[1], q)
    _required = [v for k, v in _first_of.items()]
    queries = _required + [q for q in queries if q not in _required]

    kept = []
    counts = collections.Counter()
    for q in queries:
        cand = counts.copy()
        cand[gold[q]] += 1
        n = sum(cand.values())
        if q not in _required and n >= 3 and max(cand.values()) / n > MAX_STATUS_SHARE:
            continue
        kept.append(q)
        counts[gold[q]] += 1
    if len(kept) < 2:
        return None
    queries = kept
    gold = {q: gold[q] for q in queries}
    if len(set(gold.values())) < 2:
        return None
    if len(gold) >= 3 and max(counts.values()) / len(gold) > MAX_STATUS_SHARE:
        return None
    rng.shuffle(queries)
    lines = [f"{c} under {s}: {gold[(c, s)].lower().replace('_', ' ')}" for c, s in queries]

    return SemItem(
        prompt=_render_prompt(render_ops(base), queries, ordering),
        theory_text=render_ops(base), base_ops=base, queries=queries, gold=gold,
        ordering=ordering, level=level,
        reference="[answer]\n" + "\n".join(lines) + "\n[/answer]",
        metadata={"n_items": len(base), "n_queries": len(queries),
                  "n_clusters": n_cluster, "semantics": list(sems),
                  "n_diverging_claims": len(diverging),
                  "n_rules": len(rules)})


def _render_prompt(theory: str, queries: Sequence[Tuple[str, str]], ordering: str) -> str:
    on = ("the last-link strength ordering" if ordering == LAST_LINK
          else "the weakest-link strength ordering")
    asks = "\n".join(f"   {c} under {s}" for c, s in queries)
    return (f"The following is a defeasible argumentation theory, evaluated with {on}.\n\n"
            f"{theory}\n\n"
            "State the status of each claim UNDER THE SEMANTICS NAMED BESIDE IT:\n"
            f"{asks}\n\n"
            "Possible statuses: justified, overruled, undecided. Under stable semantics, if the "
            "theory has no stable extension, answer `no stable extension`.\n\n"
            "Answer format: one line per query, written as `claim under semantics: status`, "
            "between [answer] and [/answer].")


_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)
_PAIR = re.compile(r"(-?\w+)\s+under\s+([a-z ]+?)\s*[:=]\s*"
                   r"(justified|overruled|undecided|no stable extension)\b(?!\w)", re.I)


def score(answer_text: str, item: SemItem) -> Dict:
    diag: Dict = {"n_gold": len(item.gold), "wrong": [], "missing": []}
    m = _ANSWER.search(answer_text or "")
    if m is None:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    body = m.group(1)
    pred: Dict[Tuple[str, str], str] = {}
    for claim, sem, st in _PAIR.findall(body):
        pred[(claim, sem.strip().lower())] = st.upper().replace(" ", "_")
    residue = _PAIR.sub(" ", body)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    if junk:
        diag["junk_tokens"] = junk[:6]
        return {"score": 0.0, "reason": f"unparseable_tokens:{len(junk)}", "diagnostics": diag}
    if not pred:
        return {"score": 0.0, "reason": "no_parseable_pairs", "diagnostics": diag}

    gold_pairs = {(c, s.lower(), v) for (c, s), v in item.gold.items()}
    pred_pairs = {(c, s, v) for (c, s), v in pred.items()}
    tp = len(gold_pairs & pred_pairs)
    precision = tp / max(len(pred_pairs), 1)
    recall = tp / max(len(gold_pairs), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    diag["wrong"] = sorted(f"{c} under {s}: said {v}, is {item.gold[(c, s)]}"
                           for (c, s), v in pred.items()
                           if (c, s) in item.gold and item.gold[(c, s)] != v)[:6]
    diag["n_correct"] = tp
    return {"score": round(f1, 4), "reason": "ok", "f1": round(f1, 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "exact_match": gold_pairs == pred_pairs, "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              tries: int = 24) -> Optional[SemItem]:
    for k in range(tries):
        it = build(level, seed * 83 + k, ordering)
        if it is not None:
            return it
    return None
