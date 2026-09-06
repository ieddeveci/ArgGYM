from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import UNSATISFIABLE, Operation
from arggym.core.answers import AnswerTemplate, ScoreResult, UnparseableAnswer
from arggym.core.invariants import randomize_rule_names, split_atoms_and_rules
from arggym.core.pairs import collect, pair_f1
from arggym.core.prompting import answer_format
from arggym.core.spec import ALL_ORDERINGS

TASK = "semantics_query"
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
EASY_LEVELS = 2  # every exported level must contain a claim the semantics disagree about
MAX_DIRECTIVES = 26

MAX_EAGER_ARGUMENTS = 32
MAX_STATUS_SHARE = 0.45

MIN_QUERIES = 4
MAX_QUERIES_BY_LEVEL = 12
PANEL_THRESHOLD = round(MAX_STATUS_SHARE + 0.03, 3)
_L = "abcdefghijklmnopqrstuvwxy"

GROUNDED = "grounded"
SCEPT_PREF = "sceptical preferred"
CRED_PREF = "credulous preferred"
STABLE = "stable"
EAGER = "eager"

# Additive: each level keeps every semantics the level below it had. `status_under`
# implements all five, but only two were ever scheduled, so sceptical preferred,
# credulous preferred and stable reached no item at any level, and level 3 asked one
# semantics and was `status_query` reworded (#36).
SEMANTICS_BY_LEVEL = {
    1: (GROUNDED,),
    3: (GROUNDED, CRED_PREF),
    4: (GROUNDED, CRED_PREF, EAGER),
    6: (GROUNDED, CRED_PREF, STABLE, EAGER),
    9: (GROUNDED, SCEPT_PREF, CRED_PREF, STABLE, EAGER),
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


def wants_unshielded_ring(level: int, ordering: str) -> bool:
    """Does this cell load the ring onto both branches, leaving no stable extension?

    `_render_prompt` tells every item that asks about stable to answer `no stable
    extension` if the theory has none. Before #77 no theory ever had none, so the sentence
    named an answer that was never right -- a standing red herring. A few cells have to
    make it right, and only a few: an unattacked ring answers `no stable extension` for
    every claim in the item, so on many cells that one answer would be most of what
    `stable` says, and a solver could score it without reading a theory.

    Keyed on (level, ordering), which are the item's coordinates in the exported grid, and
    not on the seed. `tasksets/standard.yaml` scans up to 40 seeds a cell and keeps the
    first two that build, so which seeds a cell ships is not fixed and a seed-keyed rule
    would realise an uncontrolled fraction. Level alone would confound the phenomenon with
    the curriculum axis the report breaks out, so the ordering carries it: one ordering
    per level, a different one at each level, no ordering odd everywhere. The exported
    levels step by 3, which is what makes `level // 3` a bijection onto the four
    orderings across them.
    """
    if STABLE not in semantics_for(level):
        return False
    return ALL_ORDERINGS[(level // 3) % len(ALL_ORDERINGS)] == ordering


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
    grounded: Optional[Dict[str, str]] = None
    preferred: Optional[List[set]] = None
    stable: Optional[List[set]] = None
    eager: Optional[set] = None


def semantics_cache(ops: Sequence[Operation], sems: Sequence[str], ordering: str,
                    verifier: Optional[ASPICVerifier] = None) -> Optional[SemCache]:
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


def _ring(host: str, names: Sequence[str]) -> List[Operation]:
    """`d0: host => -d1`, `d1: host => -d2`, ... `dn: host => -d0`.

    Undercuts, because an odd cycle needs attacks that do not answer back and an undercut
    is one: the argument for `-d` defeats every argument using rule `d`, and nothing about
    `d` reaches back. A rebut is symmetric, so a ring of rebuts is a ring of two-cycles.
    Every rule stands on `host`, so defeating the argument for `host` collapses the ring.
    """
    return [Operation(kind="defeasible", name=n, antecedents=(host,),
                      consequent="-" + names[(i + 1) % len(names)])
            for i, n in enumerate(names)]


def _odd_cycle(it, ridx, rng, unshielded: bool) -> Tuple[List[Operation], List[str]]:
    """A contested pair with an odd undercut ring standing on one side of it.

    This is the cluster that makes `stable` a different question from `sceptical
    preferred`. The two names read the same all/any/none formula off their own extension
    family, so they answer differently only where the families differ, and no cluster
    built anything that separated them: preferred equalled stable on 40 of 40 exported
    cells and 360 of 360 in a wider sweep (#77).

    Odd cycles are the whole of the difference. Dung's coherence result says an argument
    graph with no odd-length cycle has its preferred and stable extensions equal, so
    without one there is nothing for `stable` to say that `sceptical preferred` does not.

    What #77 asked for was an unattacked ring, which kills every stable extension and
    makes every claim in the theory answer `no stable extension`. Shielding the ring is
    better. With `p` and `-p` in a two-cycle and the ring standing on `p`:

        premise a          premise b
        r1: a => p         r2: b => -p
        d0: p => -d1       d1: p => -d2      d2: p => -d0

    the extension that takes `-p` defeats the argument for `p`, and with it every rule in
    the ring, so it is stable; the extension that takes `p` keeps the ring alive and is
    preferred but not stable. Two preferred extensions, one stable, on all four
    orderings. `p` is undecided under sceptical preferred and overruled under stable;
    `-p` is undecided under sceptical preferred and justified under stable. Both
    separations survive #36's rereading of overruled, since `-p` is in every stable
    extension and `p` is in one preferred extension of two either way.

    Loading a ring onto both branches instead of one gives the unattacked case back: no
    stable extension at all, and `no stable extension` for every claim. That variant is
    scheduled thinly by `wants_unshielded_ring`, because the prompt promises the answer
    on every item that asks about stable and it would otherwise never be right.
    """
    host_first = rng.random() < 0.5
    asserted_pair = rng.random() < 0.5
    # Any odd length works; 3 and 5 both to keep the ring from being one countable shape.
    # The unattacked variant carries two rings and is the widest cluster the generator
    # has, so it stays at 3 rather than spending four more of the 26 directives.
    ring_len = 3 if unshielded else rng.choice((3, 5))
    p = next(it)
    lits = [p, "-" + p]

    if asserted_pair:
        # The pair asserted outright. Cheaper by two directives than deriving it, and a
        # different surface for the same two-cycle, so the item is not one recognisable
        # shape. No preference is declared: one would settle the conflict and leave a
        # single extension, and the cluster needs both.
        ops = [Operation(kind="premise", content=p), Operation(kind="premise", content="-" + p)]
    else:
        a, b = next(it), next(it)
        ops = [Operation(kind="premise", content=a), Operation(kind="premise", content=b)]
        for src, concl in ((a, p), (b, "-" + p)):
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                 antecedents=(src,), consequent=concl))

    hosts = lits if unshielded else [lits[0] if host_first else lits[1]]
    for host in hosts:
        names = []
        for _ in range(ring_len):
            ridx[0] += 1
            names.append(f"r_{ridx[0]}")
        ops.extend(_ring(host, names))
    return ops, lits


def _strict_axiom(it, ridx):
    from arggym.aspic.engine import Operation
    ax, mid, out = next(it), next(it), next(it)
    src = next(it)
    ridx[0] += 1
    s1 = f"r_{ridx[0]}"
    ridx[0] += 1
    d1 = f"r_{ridx[0]}"
    ops = [Operation(kind="axiom", content=ax),
           Operation(kind="strict", name=s1, antecedents=(ax,), consequent=mid),
           Operation(kind="premise", content=src),
           Operation(kind="defeasible", name=d1, antecedents=(src,), consequent="-" + mid)]
    ridx[0] += 1
    d2 = f"r_{ridx[0]}"
    ops.append(Operation(kind="defeasible", name=d2, antecedents=(mid,), consequent=out))
    return ops, out


def _negated_premise(it, ridx):
    from arggym.aspic.engine import Operation
    base, out = next(it), next(it)
    ridx[0] += 1
    r = f"r_{ridx[0]}"
    ops = [Operation(kind="premise", content=base),
           Operation(kind="premise", content="-" + base),
           Operation(kind="prefer_premise", stronger="-" + base, weaker=base),
           Operation(kind="defeasible", name=r, antecedents=("-" + base,), consequent=out)]
    return ops, out


def build(level: int, seed: int, ordering: str = LAST_LINK,
          template: Optional[AnswerTemplate] = None) -> Optional[SemItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "sem"))
    sems = semantics_for(level)
    n_cluster = 2

    it = iter(_names(stable_seed(seed, level, ordering, "nm"), 40 + n_cluster * 12))
    ops: List[Operation] = []
    ridx = [0]
    candidates: List[str] = []

    _lo = 3 + (1 if level >= 6 else 0)
    _hi = min(6, _lo + 1 + (1 if level >= 11 else 0))
    _n_cluster = rng.randint(_lo, _hi)
    # The odd-cycle cluster takes a fixed slot on every item that asks about stable,
    # rather than a seventh seat on the menu. It is the only thing in the generator that
    # separates stable from sceptical preferred, and a menu draw would leave most items
    # unable to tell the two apart at all (#77).
    _odd_slot = 1 if STABLE in sems else -1
    _unshielded = wants_unshielded_ring(level, ordering)
    _required_claims: List[str] = []
    for k in range(_n_cluster):
        _menu = ["defeated", "settled", "undermining", "junction",
                 "strict_axiom", "negated_premise"]
        if k == 0:
            kind = "floating"
        elif k == _odd_slot:
            kind = "odd"
        else:
            kind = _menu[rng.randrange(len(_menu))]
        if kind == "odd":
            block, lits = _odd_cycle(it, ridx, rng, unshielded=_unshielded)
            candidates.extend(lits)
            _required_claims.extend(lits)
        elif kind == "floating":
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
        elif kind == "strict_axiom":
            block, x = _strict_axiom(it, ridx)
            candidates.append(x)
        elif kind == "negated_premise":
            block, x = _negated_premise(it, ridx)
            candidates.append(x)
        else:
            # `_odd` sat behind an `else` that the menu above could not reach, and shipped
            # unbuilt for as long as it existed (#77). A kind the menu names and the
            # dispatch forgets now stops the build instead of vanishing into a fallback.
            raise ValueError(f"cluster kind {kind!r} is on the menu and not built")
        ops.extend(block)

    # The cluster sits at slot 1, which exists only because the cluster-count floor is 4
    # at the levels that ask about stable. Lower that floor and the cluster would go
    # missing, silently, and stable would be a duplicate column again. Refusing the build
    # makes it a failure rather than a property nobody notices (#77).
    if (STABLE in sems) != bool(_required_claims):
        raise ValueError(f"level {level} asks about stable and built no odd cycle")

    if len(ops) > MAX_DIRECTIVES:
        return None

    # The eager padding is part of the theory the prompt shows, so it is built here,
    # before the rename, and renamed with everything else. Otherwise its rules keep
    # their r_<n> names beside the pool names and mark out the padding by shape alone.
    # It is held aside rather than spliced into ops so the shuffle below draws exactly
    # as it does without padding: the gates further down depend on the rng, and the
    # padding must not re-roll which items they accept.
    _pad: List[Operation] = []
    if EAGER in sems:
        _ordered = ([o for o in ops if o.kind in ("premise", "axiom")]
                    + [o for o in ops if o.kind in ("defeasible", "strict")]
                    + [o for o in ops if o.kind.startswith("prefer")])
        try:
            _na = len(ASPICVerifier.from_operations(
                _ordered, ordering=ordering).fw.af.arguments)
        except Exception:
            return None
        _fill = 0
        while _na + 2 <= MAX_EAGER_ARGUMENTS and _fill < 16:
            _a, _c = next(it), next(it)
            ridx[0] += 1
            _pad.append(Operation(kind="premise", content=_a))
            _pad.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                  antecedents=(_a,), consequent=_c))
            candidates.append(_c)
            _na += 2
            _fill += 1

    _n = len(ops)
    ops, _map = randomize_rule_names(list(ops) + _pad,
                                     stable_seed(seed, level, ordering, "rn"))
    ops, _pad = ops[:_n], ops[_n:]
    atoms, rnames = split_atoms_and_rules(ops + _pad)
    if atoms & rnames:
        return None

    rng.shuffle(ops)
    facts = [o for o in ops if o.kind in ("premise", "axiom")]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    prefs = [o for o in ops if o.kind.startswith("prefer")]
    base = facts + rules + prefs + _pad

    _cache_v: Optional[ASPICVerifier] = None
    if EAGER in sems:
        try:
            _v = ASPICVerifier.from_operations(list(base), ordering=ordering)
        except Exception:
            return None
        if len(_v.fw.af.arguments) > MAX_EAGER_ARGUMENTS:
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

    _div = set(diverging)
    by_status = collections.defaultdict(list)
    for k, v in gold.items():
        by_status[v].append(k)
    for v in by_status:
        by_status[v].sort(key=lambda q: (q[0] not in _div, q))
    for v in by_status:
        _d = [q for q in by_status[v] if q[0] in _div]
        _n = [q for q in by_status[v] if q[0] not in _div]
        rng.shuffle(_d)
        rng.shuffle(_n)
        by_status[v] = _n + _d
    order = sorted(by_status, key=lambda v: -len(by_status[v]))
    queries = []
    while any(by_status[v] for v in order):
        for v in order:
            if by_status[v]:
                queries.append(by_status[v].pop())
    # The odd cycle separates stable from sceptical preferred in the theory whatever the
    # item asks, but a theory that separates them and an item that shows it are different
    # things: the trim below keeps a fraction of the pool and rarely draws one claim under
    # both names, so the separation reached 6 of 24 cells while the theory carried it on
    # 32 of 32. The cluster's two literals are therefore asked under both names outright,
    # and asked first, so the semantics-coverage pass below fills around them rather than
    # spending the stable slot on some other claim. Required by construction, not by
    # inspection -- the cluster separates the two on all four orderings whatever the rest
    # of the theory does, so nothing here reads the status it is selecting for.
    #
    # On an unattacked ring `no stable extension` is a fact about the theory rather than
    # about a claim, so a second stable query repeats the first. One of the pair is asked
    # under stable there; both are still asked under sceptical preferred, which is where
    # the contrast shows.
    _required: List[Tuple[str, str]] = []
    _stable_lits = ([rng.choice(_required_claims)] if _unshielded and _required_claims
                    else _required_claims)
    for _s, _lits in ((STABLE, _stable_lits), (SCEPT_PREF, _required_claims)):
        if _s not in sems:
            continue
        for _c in _lits:
            if (_c, _s) in gold and (_c, _s) not in _required:
                _required.append((_c, _s))
    # Every semantics the level schedules is asked at least once, whatever the trim does.
    _covered = {q[1] for q in _required}
    for q in queries:
        if q[1] not in _covered:
            _covered.add(q[1])
            _required.append(q)
    queries = _required + [q for q in queries if q not in _required]

    _pool_n = len(queries)
    _frac = 0.55 + rng.random() * 0.35
    _target = max(MIN_QUERIES, len(_required),
                  min(MAX_QUERIES_BY_LEVEL, _pool_n, int(round(_pool_n * _frac))))
    kept = []
    counts = collections.Counter()
    _n_stable = 0
    for q in queries:
        if len(kept) >= _target:
            break
        # Same reason as the required list above: on an unattacked ring every stable
        # query has the same answer, and left uncapped the interleave below treats them
        # as a bucket to draw from and hands one item four of them.
        if _unshielded and q[1] == STABLE and _n_stable >= 1:
            continue
        cand = counts.copy()
        cand[gold[q]] += 1
        n = sum(cand.values())
        if q not in _required and n >= 6 and max(cand.values()) / n > MAX_STATUS_SHARE:
            continue
        kept.append(q)
        counts[gold[q]] += 1
        _n_stable += q[1] == STABLE
    if len(kept) < MIN_QUERIES:
        return None
    queries = kept
    gold = {q: gold[q] for q in queries}
    if len(set(gold.values())) < 2:
        return None
    if len(gold) >= 6 and max(counts.values()) / len(gold) > MAX_STATUS_SHARE:
        return None
    rng.shuffle(queries)
    lines = [f"{c} under {s}: {gold[(c, s)].lower().replace('_', ' ')}" for c, s in queries]

    return SemItem(
        prompt=_render_prompt(render_ops(base), queries, ordering, template),
        theory_text=render_ops(base), base_ops=base, queries=queries, gold=gold,
        ordering=ordering, level=level,
        reference="\n".join(lines),
        metadata={"n_items": len(base), "n_queries": len(queries),
                  "n_clusters": n_cluster, "semantics": list(sems),
                  "n_diverging_claims": len(diverging),
                  "odd_cycle": bool(_required_claims),
                  "unshielded_ring": _unshielded,
                  "n_rules": len(rules)})


def _render_prompt(theory: str, queries: Sequence[Tuple[str, str]], ordering: str,
                   template: Optional[AnswerTemplate] = None) -> str:
    on = _ordering_phrase(ordering)
    asks = "\n".join(f"   {c} under {s}" for c, s in queries)
    return (f"The following is a defeasible argumentation theory, evaluated with {on}.\n\n"
            f"{theory}\n\n"
            "State the status of each claim UNDER THE SEMANTICS NAMED BESIDE IT:\n"
            f"{asks}\n\n"
            "Possible statuses: justified, overruled, undecided.\n"
            + ("Under stable semantics, if the theory has no stable extension, answer "
               "`no stable extension`.\n" if any(s == STABLE for _, s in queries) else "")
            + "\n"
            + answer_format("Answer format: one line per query, written as "
                            "`claim under semantics: status`.", template))


# One whole line, as the format clause above asks for (`core/pairs.py`).
_PAIR = re.compile(r"(-?\w+)\s+under\s+([a-z ]+?)\s*[:=]\s*"
                   r"(justified|overruled|undecided|no stable extension)\b(?!\w)", re.I)


#: A query -- claim and semantics -- keyed to the status the answer gives it. A sequence
#: where the answer gave several statuses for one query: only text can contradict itself,
#: so a solver that submits a mapping writes one status per query and a text answer that
#: hedges is the only way to reach the `contradicted` diagnostic (`core/pairs.py:pair_f1`).
Value = Dict[Tuple[str, str], Union[str, Sequence[str]]]


def _key(claim: str, semantics: str) -> Tuple[str, str]:
    """`x under grounded` and `X under grounded semantics` ask the same query."""
    return claim.lower(), re.sub(r"\s+semantics$", "", semantics.strip().lower())


def _diagnostics(item: SemItem) -> Dict:
    return {"n_gold": len(item.gold), "wrong": [], "missing": [], "contradicted": []}


def parse(answer_text: str, item: SemItem) -> Value:
    """The query-status pairs the answer states, in the order it states them.

    A status repeated for one query is kept rather than folded away here: whether saying
    the same thing twice is one prediction is a scoring question, and `score_value`
    answers it.
    """
    body = answer_text or ""
    said: Dict[Tuple[str, str], List[str]] = {}
    for claim, sem, st in _PAIR.findall(body):
        said.setdefault(_key(claim, sem), []).append(st.upper().replace(" ", "_"))
    residue = _PAIR.sub(" ", body)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    if junk:
        raise UnparseableAnswer(f"unparseable_tokens:{len(junk)}", {"junk_tokens": junk[:6]})
    if not said:
        raise UnparseableAnswer("no_parseable_pairs")
    return said


def score_value(value: Value, item: SemItem) -> ScoreResult:
    """Grade a label map, whichever way it arrived.

    A solver with constrained decoding, a JSON schema or a tool call submits one of these
    and never imitates the line format (`docs/dataset-contract.md` section 4).
    """
    diag = _diagnostics(item)
    pred = collect((_key(claim, sem), st.upper().replace(" ", "_"))
                   for (claim, sem), statuses in value.items()
                   for st in ((statuses,) if isinstance(statuses, str) else statuses))

    gold = {(c, s.lower()): v for (c, s), v in item.gold.items()}
    r = pair_f1(pred, gold)
    diag["wrong"] = sorted(f"{c} under {s}: said {'/'.join(v)}, is {gold[(c, s)]}"
                           for (c, s), v in pred.items()
                           if (c, s) in gold and v != [gold[(c, s)]])[:6]
    diag["missing"] = sorted(f"{c} under {s}" for (c, s) in gold if (c, s) not in pred)[:6]
    diag["contradicted"] = [f"{c} under {s}" for c, s in r.contradicted][:6]
    diag["n_correct"] = r.tp
    diag.update(f1=round(r.f1, 4), precision=round(r.precision, 4),
                recall=round(r.recall, 4), exact_match=r.exact_match)
    return ScoreResult(round(r.f1, 4), r.exact_match, "ok", diag)


def score(answer_text: str, item: SemItem) -> ScoreResult:
    """Grade an answer written as text: read the value out of it, then grade the value.

    The text arrives already extracted: composing the prompt and pulling the answer out
    of whatever came back is the harness's job, so the dataset never unwraps a fence.
    That is what lets a caller use any convention at all -- or none, with a solver that
    submits the value itself (`docs/dataset-contract.md` sections 1 and 4).
    """
    try:
        value = parse(answer_text, item)
    except UnparseableAnswer as bad:
        return ScoreResult(0.0, False, bad.reason, {**_diagnostics(item), **bad.diagnostics})
    return score_value(value, item)


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              tries: int = 24,
              template: Optional[AnswerTemplate] = None) -> Optional[SemItem]:
    for k in range(tries):
        it = build(level, seed * 83 + k, ordering, template)
        if it is not None:
            return it
    return None
