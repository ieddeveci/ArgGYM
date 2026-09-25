from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import UNSATISFIABLE, Operation
from arggym.core.answers import ScoreResult, UnparseableAnswer
from arggym.core.build import BuildReport, Rejected, retry
from arggym.core.invariants import randomize_rule_names, split_atoms_and_rules
from arggym.core.pairs import collect, pair_f1
from arggym.core.prompting import STRAY_TEXT, answer_format
from arggym.core.scoring import unmark
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


MAX_DIRECTIVES = 26

MAX_EAGER_ARGUMENTS = 32
# The most of one semantics' queries in an item that one status may take, wherever the
# item asks that semantics `MIN_ASKS_FOR_SHARE` times or more. The trim below enforces it
# and the row it ships is checked again. It is counted per semantics because a count over
# the whole item can balance while each semantics answers one word: counted that way, one
# constant per semantics, reading no theory, scored 0.70 on the release grid (#104).
#
# The count starts at four asks because a cap that binds on two or three decides them: at
# two, the second answer has to differ from the first; at three, the answers have to be a
# permutation of the three statuses. A reader who solves one query of such a set is handed
# its partners. From four up the cap leaves several patterns open.
#
# There is no second cap over the item's pooled answers. With this one binding only from
# four asks, a pooled cap at the same share refuses so many candidates that a cell on each
# grid falls under `min_build_acceptance`.
MAX_STATUS_SHARE = 0.5
MIN_ASKS_FOR_SHARE = 4

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
#
# Levels 1, 2 and 3 are one band rather than a ramp. Three of `build`'s level-dependent
# decisions agree across the three -- the cluster count is `randint(3, 4)` below level 6,
# the unshielded ring needs stable, and the semantics are this pair -- so only the seed
# salt separates them and the three levels are re-seedings of one generator. A theory
# runs 14 to 24 directives on all of them (40 seeds x 4 orderings a level, and again at
# 200). The first rung the task has is level 4. `data/taskset.yaml` ships levels 1
# to 3 because it ships every level, not because they are easier.
#
# Credulous preferred is the second member because the floating cluster every item opens
# with separates it from grounded by construction -- the contested literal and the
# floating conclusion above it are undecided under grounded and justified under credulous
# preferred. Measured on every built item over levels 1-15 and all four orderings, both
# literals split on 1509 of 1509. Eager was the alternative and is not eligible at the
# entry levels: the padding below fills the theory out to `MAX_EAGER_ARGUMENTS`, which
# takes those 14-24 directives to 31-38, so asking it at level 1 would put the task's
# largest theories on its smallest level (#108).
SEMANTICS_BY_LEVEL = {
    1: (GROUNDED, CRED_PREF),
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
    not on the seed. `data/taskset.yaml` scans up to 40 seeds a cell and keeps the
    first `take` that build, so which seeds a cell ships is not fixed and a seed-keyed rule
    would realise an uncontrolled fraction.

    One ordering per stable-asking level, so a ring is on a quarter of the rows that ask
    about stable and no more. The rotation walks the four orderings one level at a time,
    so each ordering's ring levels are four apart rather than sitting in one block, where a per-ordering `stable` mean would read level difficulty
    as an ordering effect (#168). The stable-asking levels are a contiguous run --
    `semantics_for` only ever adds semantics as the level rises -- and any contiguous run
    of levels lands within one of balanced: the ten that ask today split 3/3/2/2.
    tests/test_stable_says_what_sceptical_preferred_cannot.py pins the table.
    """
    if STABLE not in semantics_for(level):
        return False
    return ALL_ORDERINGS[level % len(ALL_ORDERINGS)] == ordering


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
    """A contested pair with an odd undercut ring standing over one side of it.

    This is the cluster that makes `stable` a different question from `sceptical
    preferred`. The two names read the same all/any/none formula off their own extension
    family, so they answer differently only where the families differ, and nothing the
    generator built made them differ: preferred equalled stable on all 40 cells the grid
    exported then, and 360 of 360 in a wider sweep (#77).

    Odd cycles are the whole of the difference. Dung's coherence result says an argument
    graph with no odd-length cycle has its preferred and stable extensions equal, so
    without one there is nothing for `stable` to say that `sceptical preferred` does not.

    What #77 asked for was an unattacked ring, which kills every stable extension and
    makes every claim in the theory answer `no stable extension`. Shielding the ring is
    better. With `p` and `-p` asserted against each other and the ring hung from `q`:

        premise p          premise -p         rq: p => q
        d0: q => -d1       d1: q => -d2       d2: q => -d0

    the extension taking `-p` defeats the argument for `p`, so for `q`, so every rule in
    the ring, and is stable; the extension taking `p` keeps the ring alive and is preferred
    but not stable. Two preferred extensions, one stable, on all four orderings. Both
    literals are undecided under sceptical preferred; stable justifies one and overrules
    the other. Both separations survive #36's rereading of overruled, since the loser's
    contrary is in every stable extension and each literal is in one preferred extension of
    two either way.

    Loading a ring over both branches gives the unattacked case back: no stable extension
    at all, and `no stable extension` for every claim. `wants_unshielded_ring` schedules
    that thinly, because the prompt promises the answer on every item that asks about
    stable and it would otherwise never be right.

    Three things here keep the cluster from handing a solver more than it has to. None of
    them makes the item hard to fake: a program that reads the ring off the theory text and
    computes no extension answers most of the stable column, and that is a property of odd
    cycles rather than of this code. `docs/dataset-card.md` carries the measurement, and
    #95 tracks what it means for chance floors on a task that asks about five semantics.

    * One literal is exposed as a query candidate, not both. Asking both put a
      complementary pair in every item under sceptical preferred, where the cluster makes
      both undecided by construction, so `if c and -c are both asked, answer undecided`
      covered that column and was never wrong. Trading #77's duplicated column for a
      guessable neighbouring one is no trade.
    * The ring hangs from a literal the branch derives rather than from the branch itself,
      so the literal the item asks about is not the one the ring names, and reading the
      ring off the text answers the query only after tracing a rule back to it.
    * The pair is asserted outright rather than derived through two rules, and the ring is
      three long rather than three or five. Six directives against nine, which matters
      because this cluster is on every item that asks about stable and the theory is capped
      at MAX_DIRECTIVES. Neither longer form separates the semantics any better, so
      neither earns its directives.
    """
    p = next(it)
    lits = [p, "-" + p]
    ops = [Operation(kind="premise", content=p),
           Operation(kind="premise", content="-" + p)]
    for branch in (lits if unshielded else [rng.choice(lits)]):
        # The unattacked variant skips the carrier: its two rings sit over complementary
        # literals, which gives the case away whether or not they are one step removed,
        # and it is already the widest cluster here against MAX_DIRECTIVES.
        host = branch
        if not unshielded:
            ridx[0] += 1
            host = next(it)
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                 antecedents=(branch,), consequent=host))
        names = []
        for _ in range(3):
            ridx[0] += 1
            names.append(f"r_{ridx[0]}")
        ops.extend(_ring(host, names))
    return ops, [rng.choice(lits)]


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


def build(level: int, seed: int, ordering: str = LAST_LINK) -> Union[SemItem, Rejected]:
    rng = random.Random(stable_seed(seed, level, ordering, "sem"))
    sems = semantics_for(level)
    # The name pool's size, not the theory's cluster count -- that is `_n_cluster` below,
    # drawn per level from the curriculum. The two were a character apart and the metadata
    # shipped this one, so `n_clusters` was the constant 2 on every row of the grid, a
    # value the draw cannot produce at any level (#141).
    _NAME_POOL_CLUSTERS = 2

    it = iter(_names(stable_seed(seed, level, ordering, "nm"),
                     40 + _NAME_POOL_CLUSTERS * 12))
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
        else:
            block, x = _negated_premise(it, ridx)
            candidates.append(x)
        ops.extend(block)

    if len(ops) > MAX_DIRECTIVES:
        return Rejected("theory_over_max_directives")

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
        except Exception as exc:
            return Rejected(f"eager_argument_count_failed:{type(exc).__name__}")
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
        return Rejected("atom_rule_name_collision")

    rng.shuffle(ops)
    facts = [o for o in ops if o.kind in ("premise", "axiom")]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    prefs = [o for o in ops if o.kind.startswith("prefer")]
    # Every padding claim is justified, so the padding has to be listed among the other
    # facts and rules: appended after the preferences, it marked its claims as justified
    # by position alone (#208). Its own rng keeps the main stream where it was.
    listed_facts, listed_rules = facts, rules
    if _pad:
        _prng = random.Random(stable_seed(seed, level, ordering, "pad"))
        listed_facts = facts + [o for o in _pad if o.kind == "premise"]
        listed_rules = rules + [o for o in _pad if o.kind == "defeasible"]
        _prng.shuffle(listed_facts)
        _prng.shuffle(listed_rules)
    base = listed_facts + listed_rules + prefs

    _cache_v: Optional[ASPICVerifier] = None
    if EAGER in sems:
        try:
            _v = ASPICVerifier.from_operations(list(base), ordering=ordering)
        except Exception as exc:
            return Rejected(f"eager_verifier_error:{type(exc).__name__}")
        if len(_v.fw.af.arguments) > MAX_EAGER_ARGUMENTS:
            return Rejected("too_many_eager_arguments")
        _cache_v = _v

    cache = semantics_cache(base, sems, ordering, verifier=_cache_v)
    if cache is None:
        return Rejected("semantics_cache_failed")
    gold: Dict[Tuple[str, str], str] = {}
    for c in candidates:
        for s in sems:
            # The odd cycle's literals are never asked under grounded or credulous
            # preferred. It is a contested pair, so those two answer it by construction --
            # undecided and justified respectively, on every item -- and asking them there
            # hands a solver two free answers and tilts two columns the cluster has no
            # business tilting. Structural, so it costs no look at the gold it drops (#77).
            if c in _required_claims and s in (GROUNDED, CRED_PREF):
                continue
            st = status_under(cache, c, s)
            if st is None:
                return Rejected("status_unavailable")
            gold[(c, s)] = st
    if not gold:
        return Rejected("no_gold_queries")

    by_claim: Dict[str, List[str]] = {}
    for (c, s) in gold:
        by_claim.setdefault(c, []).append(s)
    diverging = [c for c in by_claim if len({gold[(c, s)] for s in by_claim[c]}) > 1]
    # Every level, not every exported level. `EASY_LEVELS = 2` exempted levels 1 and 2
    # back when they asked grounded alone and could not have a diverging claim; now that
    # they ask the pair, an item without one is `status_query` reworded wherever it lands.
    # Dropping the exemption rejects nothing the generator builds today -- 800 candidates
    # at each of levels 1 and 2 all carry at least two diverging claims -- so it only bites
    # if the floating cluster at slot 0 later stops separating the two semantics (#108).
    if not diverging:
        return Rejected("no_diverging_claim")

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
    # things: the trim below keeps a fraction of the pool and rarely drew one claim under
    # both names, so the separation reached 6 of 24 cells while the theory carried it on
    # 32 of 32. The cluster's exposed literal is therefore asked under stable outright, and
    # asked first, so the semantics-coverage pass below fills around it rather than
    # spending the stable slot on some other claim. Required by construction, not by
    # inspection -- the cluster separates the two on all four orderings whatever the rest
    # of the theory does, so nothing here reads the status it is selecting for.
    #
    # One claim is then asked under both names. Pairing the cluster's own literal on every
    # item made the pairing itself the answer: the cluster leaves both its literals in one
    # preferred extension of two, so sceptical preferred calls them undecided by
    # construction. A claim from another cluster carries the pairing when the coin below
    # says so and a decoy is available.
    #
    # The coin is fair and the realised split is not, because on a decoy item the
    # cluster's literal often survives the trim under both names anyway. Over the 280 rows
    # of the release grid that ask both names, 264 pair a claim; the cluster's own literal
    # is one of the paired claims on 202 of them and the only one on 77. So this mitigates
    # the pattern rather than removing it, and the sceptical-preferred column should be
    # read with that in mind.
    _forced: List[Tuple[str, str]] = [(c, STABLE) for c in _required_claims]
    if _required_claims:
        _decoys = [c for c in candidates if c not in _required_claims]
        _pair = (_required_claims[0] if (rng.random() < 0.5 or not _decoys)
                 else rng.choice(_decoys))
        _forced += [(_pair, STABLE), (_pair, SCEPT_PREF)]
    _required: List[Tuple[str, str]] = []
    for _c, _s in _forced:
        if _s in sems and (_c, _s) in gold and (_c, _s) not in _required:
            _required.append((_c, _s))
    # Every semantics the level schedules is asked at least once, whatever the trim does.
    #
    # Which query covers it is free, and spending it on a claim already required under a
    # different status buys a diverging claim for nothing. The coverage set asked one
    # query per semantics on mostly distinct claims, so it contained no claim answered
    # twice and therefore no divergence by construction -- which is what left ten rows
    # asking nothing two semantics disagree about (#138). Preference, not a requirement:
    # where no required claim splits, the first query covering the semantics is taken.
    _covered = {q[1] for q in _required}
    for q in queries:
        if q[1] in _covered:
            continue
        _here = [x for x in queries if x[1] == q[1] and x not in _required]
        _splits = [x for x in _here
                   if any(rc == x[0] and gold[(rc, rs)] != gold[x] for (rc, rs) in _required)]
        _covered.add(q[1])
        _required.append(_splits[0] if _splits else q)
    queries = _required + [q for q in queries if q not in _required]

    _pool_n = len(queries)
    _frac = 0.55 + rng.random() * 0.35
    _target = max(MIN_QUERIES, len(_required),
                  min(MAX_QUERIES_BY_LEVEL, _pool_n, int(round(_pool_n * _frac))))
    kept = []
    counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    _n_stable = 0
    for q in queries:
        if len(kept) >= _target:
            break
        # On an unattacked ring `no stable extension` is a fact about the theory rather
        # than about a claim, so every stable query in the item repeats the first. Left
        # uncapped the interleave above treats them as a bucket to draw from and hands one
        # item four copies of one answer. The cap runs before the required exemption below,
        # so on these items a decoy pairing keeps its sceptical-preferred query and loses
        # its stable one -- no loss, since that answer would have been the same word again.
        if _unshielded and q[1] == STABLE and _n_stable >= 1:
            continue
        # The share is counted within the query's own semantics, and it is the
        # candidate's own status that is compared, `cand[gold[q]]` rather than
        # `max(cand.values())`: a query whose status is under its share pulls the
        # semantics toward balance and is kept even while another status is over it
        # (#171). Required queries are exempt here and answer to the check after the loop.
        cand = counts[q[1]].copy()
        cand[gold[q]] += 1
        n = sum(cand.values())
        if (q not in _required and n >= MIN_ASKS_FOR_SHARE
                and cand[gold[q]] / n > MAX_STATUS_SHARE):
            continue
        kept.append(q)
        counts[q[1]][gold[q]] += 1
        _n_stable += q[1] == STABLE
    if len(kept) < MIN_QUERIES:
        return Rejected("too_few_queries_kept")
    queries = kept
    gold = {q: gold[q] for q in queries}
    if len(set(gold.values())) < 2:
        return Rejected("fewer_than_two_gold_statuses")
    for c in counts.values():
        n = sum(c.values())
        if n >= MIN_ASKS_FOR_SHARE and max(c.values()) / n > MAX_STATUS_SHARE:
            return Rejected("one_status_over_its_share_within_semantics")
    # Over the queries the row ships, not the ones it considered. The coverage preference
    # above is the mechanism and this is the assertion: the preference can find no claim
    # that splits, the trim can drop the ones it found, and a row no two semantics disagree
    # about is `status_query` with a longer answer format (#138, #36). It refuses 6 of the
    # 1256 candidates the release grid builds.
    _kept_by_claim: Dict[str, List[str]] = {}
    for (c, s) in gold:
        _kept_by_claim.setdefault(c, []).append(s)
    kept_diverging = [c for c in _kept_by_claim
                      if len({gold[(c, s)] for s in _kept_by_claim[c]}) > 1]
    if not kept_diverging:
        return Rejected("no_diverging_claim_kept")
    rng.shuffle(queries)
    lines = [f"{c} under {s}: {gold[(c, s)].lower().replace('_', ' ')}" for c, s in queries]

    return SemItem(
        prompt=_render_prompt(render_ops(base), queries, ordering),
        theory_text=render_ops(base), base_ops=base, queries=queries, gold=gold,
        ordering=ordering, level=level,
        reference="\n".join(lines),
        metadata={"n_items": len(base), "n_queries": len(queries),
                  "n_clusters": _n_cluster, "semantics": list(sems),
                  "n_diverging_claims": len(kept_diverging),
                  "odd_cycle": bool(_required_claims),
                  "odd_cycle_claims": list(_required_claims),
                  "unshielded_ring": _unshielded,
                  "n_rules": len(rules)})


def _render_prompt(theory: str, queries: Sequence[Tuple[str, str]], ordering: str) -> str:
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
                            "`claim under semantics: status`.\n" + STRAY_TEXT))


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
    body = unmark(answer_text)
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
    of whatever came back is the harness's job, so the dataset never strips the answer delimiters.
    That is what lets a caller use any convention at all -- or none, with a solver that
    submits the value itself (`docs/dataset-contract.md` sections 1 and 4).
    """
    try:
        value = parse(answer_text, item)
    except UnparseableAnswer as bad:
        return ScoreResult(0.0, False, bad.reason, {**_diagnostics(item), **bad.diagnostics})
    return score_value(value, item)


def make_item_report(level: int, seed: int, ordering: str = LAST_LINK,
                     tries: int = 24) -> BuildReport:
    return retry(lambda k: build(level, seed * 83 + k, ordering), tries)


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              tries: int = 24) -> Optional[SemItem]:
    return make_item_report(level, seed, ordering, tries).item
