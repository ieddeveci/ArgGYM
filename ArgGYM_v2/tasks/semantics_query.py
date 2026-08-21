"""semantics_query -- status under a NAMED semantics.

Every other task in the suite evaluates under grounded. Grounded is unique and maximally sceptical,
which makes it the right default for a determinate answer, but it means nothing in ArgGYM ever tests
the distinction between semantics, or between sceptical and credulous acceptance.

THE MEASURED CONSTRAINT. Preferred is intractable above roughly 25 directives, and the cost is NOT in
enumerating extensions -- it is in argument construction and the admissibility search:

    17 directives, 4 preferred extensions -> 0.20s
    22 directives, 4 preferred extensions -> 2.51s
    32 directives, 4 preferred extensions -> over 12s

So this task cannot share `status_query`'s theories, which reach 280 directives. It is a separate task
with small theories, and its curriculum grows by WHICH DISTINCTION IS REQUIRED rather than by size.

THE STRUCTURES, each verified for cost and divergence:

    floating conclusion   6 ops   f: grounded UNDECIDED, sceptical preferred JUSTIFIED
    two-cycle             6 ops   p: sceptical UNDECIDED, credulous JUSTIFIED
    odd conflict          9 ops   no stable extension in some configurations
    reinstatement         6 ops   all semantics agree -- a control, so divergence is not automatic

The floating conclusion is the important one. Both preferred extensions contain `f` -- one via `p`, one
via `-p` -- so it is sceptically justified while grounded declines to commit. This is one of the field's
longest-running disagreements, and no other ArgGYM task can express it.
"""
from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier
from core.curriculum import junction_budget, JUNCTION_CAPS
from core.invariants import randomize_rule_names, split_atoms_and_rules

TASK = "semantics_query"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
EASY_LEVELS = 3
MAX_DIRECTIVES = 26

# EAGER'S CEILING IS ARGUMENTS, NOT DIRECTIVES.
#
# Measured on the leanest possible encoding -- independent one-step rules, one argument each:
#
#     16 arguments ->  8.46s
#     18 arguments -> 42.63s
#     20 arguments -> over 90s
#
# Growth is 2.24x per additional argument, so a 60-second budget buys 18 and no more: 19 is 96s, 20 is
# 215s, and 40 would be roughly 72 years. Structure does not help -- a determinate chain with ONE
# semi-stable extension costs the same as a cyclic one at equal argument count, because semi-stable
# enumeration searches the argument powerset rather than the extensions.
#
# Only levels that ASK under eager pay this. Everything below level 14 is bounded by preferred, which
# is far cheaper, and keeps the larger MAX_DIRECTIVES.
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


def status_under(ops: Sequence[Operation], claim: str, semantics: str,
                 ordering: str) -> Optional[str]:
    """The status of `claim` under a named semantics.

    Each of the four is determinate for a given claim even where the EXTENSIONS are not, which is what
    makes per-claim scoring possible at all. The multiplicity lives in the extensions, not in the
    answer to "is this claim accepted".

    Sceptical stable on a framework with NO stable extension is vacuously true of everything. That is
    the standard convention and it is why existence must be reported separately rather than folded into
    the status.
    """
    try:
        v = ASPICVerifier.from_operations(list(ops), ordering=ordering)
    except Exception:
        return None
    try:
        if semantics == GROUNDED:
            return str(v.status(claim))
        if semantics in (SCEPT_PREF, CRED_PREF):
            exts = v.preferred_conclusions()
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
            # UNIQUE, like grounded, so the status is determinate and there is no sceptical/credulous
            # split. Eager is less sceptical: it commits where grounded declines, which is the whole
            # reason for asking under both.
            eager = {str(getattr(x, "conclusion", x)) for x in v.fw.eager_extension()}
            if claim in eager:
                return "JUSTIFIED"
            return "OVERRULED" if ("-" + claim if not claim.startswith("-")
                                   else claim[1:]) in eager else "UNDECIDED"
        if semantics == STABLE:
            exts = v.stable_conclusions()
            if not exts:
                return "NO_STABLE_EXTENSION"
            if all(claim in e for e in exts):
                return "JUSTIFIED"
            if any(claim in e for e in exts):
                return "UNDECIDED"
            return "OVERRULED"
    except Exception:
        return None
    return None


@dataclass
class SemItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    queries: List[Tuple[str, str]]          # (claim, semantics)
    gold: Dict[Tuple[str, str], str]
    ordering: str
    level: int
    reference: str
    metadata: Dict = field(default_factory=dict)


def _floating(it, ridx) -> Tuple[List[Operation], str, str]:
    """Two rival premises, both routes reaching the SAME conclusion.

    Verified: the shared conclusion is UNDECIDED under grounded and JUSTIFIED under sceptical
    preferred, because both preferred extensions contain it by different routes."""
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
    """A reinstatement chain. All four semantics agree, so divergence is not automatic."""
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
    """A loop that undermines itself: p is derived from -p by way of q.

    THE STRUCTURE WHERE EAGER AND GROUNDED PART. Grounded refuses to commit to -p or q, because the
    chain leads to p and p contradicts its own root. Eager commits: the argument for -p sits in every
    semi-stable extension and defends itself, so it is admissible throughout.

    Found by random search over small ASPIC+ theories rather than constructed from theory -- 2
    divergences in 60 tries, and this is the smallest. Purely abstract frameworks diverge far more
    readily (4 in 4000 random graphs), but ASPIC+ instantiation constrains the attack relation:
    sub-argument closure and contrariness-derived attacks rule out most of the asymmetric shapes that
    separate the two semantics.

    Verified identical under both orderings, and cheap: four such loops in sixteen directives cost
    0.33s.
    """
    from aspic.engine import Operation
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
    """A conclusion resting on TWO or THREE branches, one of which is contested.

    semantics_query had NO multi-antecedent rules at all -- its structures are built from single-
    antecedent chains. A junction whose branches disagree across extensions is exactly where the
    semantics differ: the conclusion is undecided under grounded and can be justified under preferred
    if every extension supplies some branch."""
    from aspic.engine import Operation
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
    # contest the FIRST branch, so the junction's status depends on how that contest resolves
    ops.append(Operation(kind="premise", content="-" + roots[0]))
    return ops, concl


def _defeated(it, ridx):
    """A claim decisively DEFEATED by a preference: overruled under every semantics.

    Needed because the other three structures produce only JUSTIFIED and UNDECIDED, so gold had two
    statuses and the blind-guess floor could not fall below one half. A third status is what moves the
    floor towards one third. Verified: the loser is OVERRULED under grounded, sceptical preferred,
    credulous preferred and stable alike, because the preference settles the conflict in every
    extension."""
    from aspic.engine import Operation
    a, b, x = next(it), next(it), next(it)
    ridx[0] += 1
    pro = f"r_{ridx[0]}"
    ridx[0] += 1
    con = f"r_{ridx[0]}"
    ops = [Operation(kind="premise", content=a), Operation(kind="premise", content=b),
           Operation(kind="defeasible", name=pro, antecedents=(a,), consequent=x),
           Operation(kind="defeasible", name=con, antecedents=(b,), consequent="-" + x),
           Operation(kind="prefer_rule", stronger=con, weaker=pro),
           # A PREMISE preference as well as a rule preference. Under weakest-link an argument is only
           # as strong as its weakest element, so a rule preference alone leaves the conflict
           # UNDECIDED -- verified. Without this the cluster produced one distinct status, the balance
           # check rejected the item, and levels 1-3, which ask under grounded only, had nothing left
           # to vary: yield was 0/4 under weakest-link at every level below 4.
           Operation(kind="prefer_premise", stronger=b, weaker=a)]
    return ops, x


def _odd(it, ridx) -> Tuple[List[Operation], str]:
    """Three claims in a cycle of conflict. Drives stable non-existence in some configurations."""
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
    # Clusters STOP growing at 2. Levels 12-15 add stable semantics, and adding a third
    # cluster at the same time pushed generation to 61s per item. The curriculum here is
    # meant to grow by WHICH DISTINCTION IS REQUIRED, not by theory size -- that is the
    # whole reason this is a separate task from status_query.
    n_cluster = 2

    it = iter(_names(stable_seed(seed, level, ordering, "nm"), 40 + n_cluster * 12))
    ops: List[Operation] = []
    ridx = [0]
    candidates: List[str] = []

    # The FIRST cluster is always floating. Divergence is not automatic -- a settled or odd cluster
    # alone gives an item on which every semantics agrees, which tests nothing about semantics and is
    # rejected below. Rotating freely left level 4 with no floating cluster and a 0/4 yield.
    # A DEFEATED cluster is always present alongside the floating one, so gold carries three
    # statuses rather than two. With only JUSTIFIED and UNDECIDED available the blind-guess floor is
    # one half by construction, whatever the balancing does.
    kinds = ["defeated", "settled", "odd"]
    # TWO clusters total, always floating + defeated. Adding a third pushed level 12 from
    # 0.2s to 17.6s -- preferred enumeration is exponential in the argument graph, so every
    # extra cluster is expensive. Two suffice: floating gives the grounded/preferred
    # divergence and defeated gives the third status the floor needs.
    j_budget = junction_budget(level, JUNCTION_CAPS.get("semantics_query", 3))
    # The self-undermining cluster REPLACES the junction cluster when eager is asked, rather than
    # adding to it. Every cluster adds arguments and eager is computed from the semi-stable extensions,
    # so the cost is exponential in the graph: measured, adding a fourth cluster took level 14 from
    # about 3s to 61.8s. Three clusters is the ceiling this task can afford.
    # At eager levels a FOURTH cluster fits inside the 18-argument budget: the three base clusters
    # come to about 14 arguments, and a settled chain adds three or four. Below eager levels the
    # budget is bounded by preferred instead, which is far cheaper, so a fourth cluster is affordable
    # there too.
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
            # BINARY only. A third branch adds another argument to the graph and preferred
            # enumeration is exponential in it -- measured, level 12 went from 2.9s to 17.1s with a
            # ternary junction. This task is bounded by that cost in a way no other task is, so it
            # takes the cheaper structure and the ternary arity lives in the tasks that can afford it.
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

    if EAGER in sems:
        # FILL THE ARGUMENT BUDGET EXACTLY.
        #
        # Three clusters come to 14 arguments and a fourth to 20, so cluster granularity cannot land
        # on 18. Lean units -- one premise feeding one rule -- add exactly two arguments each, so the
        # budget can be spent to the last one. Each is an independent claim, which also gives the
        # query selector more material to balance over.
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

    # every (claim, semantics) pair the item could ask, with its engine-computed answer
    gold: Dict[Tuple[str, str], str] = {}
    for c in candidates:
        for s in sems:
            st = status_under(base, c, s, ordering)
            if st is None:
                return None
            gold[(c, s)] = st
    if not gold:
        return None

    # keep the pairs, preferring ones where the semantics DISAGREE -- an item on which every
    # semantics gives the same answer tests nothing about semantics
    by_claim: Dict[str, List[str]] = {}
    for (c, s) in gold:
        by_claim.setdefault(c, []).append(s)
    diverging = [c for c in by_claim if len({gold[(c, s)] for s in by_claim[c]}) > 1]
    if level > EASY_LEVELS and not diverging:
        return None

    # BALANCED SELECTION, ported from `status_query`.
    #
    # Measured before this: gold was 61.1% JUSTIFIED and a blind "everything justified" scored 0.611
    # against this task's own 0.48 threshold. The cause is structural rather than accidental -- a
    # floating conclusion is JUSTIFIED under sceptical preferred, credulous preferred AND stable, and
    # only UNDECIDED under grounded, so asking one claim under four semantics yields three "justified"
    # answers by construction.
    #
    # `status_query` has this step for exactly the same reason and it was not carried over when this
    # task was written. Selection caps the modal share per item, which is what moves the floor from the
    # majority class down towards the number of distinct answers.
    by_status = collections.defaultdict(list)
    for k, v in gold.items():
        by_status[v].append(k)
    for v in by_status:
        by_status[v].sort()
        rng.shuffle(by_status[v])
    order = sorted(by_status, key=lambda v: -len(by_status[v]))
    queries = []
    # round-robin across statuses, so the scarcest is never crowded out
    while any(by_status[v] for v in order):
        for v in order:
            if by_status[v]:
                queries.append(by_status[v].pop())
    # GUARANTEE REPRESENTATION FOR EVERY SEMANTICS ASKED. Balancing alone selects on status, so at
    # eager levels -- where five semantics compete for three query slots -- eager pairs were dropped
    # entirely: measured 0 eager queries across generated items. A semantics the item pays for and
    # never asks about is wasted cost, and eager is the most expensive of the five.
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
        # required pairs bypass the share cap; the cap then governs everything after them
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
