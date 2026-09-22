from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.core.answers import ScoreResult, UnparseableAnswer
from arggym.core.build import BuildReport, Rejected, retry
from arggym.core.curriculum import JUNCTION_CAPS, PROFILES, junction_budget
from arggym.core.nlforms import (
    AXIOM,
    DEFEASIBLE,
    FORWARD_CONNECTIVES,
    JUNCTION_DEFEASIBLE,
    JUNCTION_STRICT,
    LINE_TRANSITIONS,
    NEGATED_AXIOM,
    NEGATED_LITERAL,
    NEGATED_PREMISE,
    NEW_TOPIC,
    OBJECTION_CONNECTIVES,
    OPENERS,
    PREFER_PREMISE,
    PREFER_RULE,
    PREMISE,
    REBUT_RULE,
    RULE_ANAPHORA,
    RULE_REFERENCE,
    STRICT,
    STRICT_EXCLUSION,
    STRICT_FROM_NEGATION,
    UNDERCUT,
)
from arggym.core.prompting import answer_format, formalization_notation
from arggym.core.scoring import parse_answer

TASK = "formalization"
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
EASY_LEVELS = 3
_L = "abcdefghijklmnopqrstuvwxy"


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def _names(seed: int, n: int) -> List[str]:
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:14] for b in _L[10:] for d in range(10)]
    rng.shuffle(pool)
    return pool[:n]


def status_map(ops: Sequence[Operation], lits: Sequence[str], ordering: str) -> Dict[str, str]:
    try:
        v = ASPICVerifier.from_operations(list(ops), ordering=ordering)
        return {l: str(v.status(l)) for l in lits}
    except Exception:
        return {}


def render_op(o: Operation) -> str:
    if o.kind in ("premise", "axiom"):
        return f"[{o.kind}: {o.content}]"
    if o.kind in ("defeasible", "strict"):
        arrow = "=>" if o.kind == "defeasible" else "->"
        return f"[{o.kind} {o.name}: {' AND '.join(o.antecedents)} {arrow} {o.consequent}]"
    return f"[{o.kind}: {o.stronger} > {o.weaker}]"


@dataclass
class FItem:
    prompt: str
    nl_text: str
    reference_ops: List[Operation]
    reference: str
    queried: List[str]
    gold_status: Dict[str, str]
    ordering: str
    level: int
    metadata: Dict = field(default_factory=dict)


_LEAD_INS = ["Note that ", "Observe that ", "It is also stated that ", "We are told that ",
             "Here, ", "Now, "]
_ATOM_START = re.compile(r"^[a-z]{2}\d")


_SELF_CONNECTED = ("in the absence", "given ", "presumably", "from ", "normally", "typically",
                   "there is no case")


def _needs_connective(text: str) -> bool:
    t = text.strip().lower()
    return not any(t.startswith(x) for x in _SELF_CONNECTED)


def _sentence(text: str, rng: random.Random, prefix: str = "",
              at_start: bool = False) -> str:
    text = text.strip().rstrip(".").strip()
    if not text:
        return ""
    if prefix and _needs_connective(text):
        return prefix[0].upper() + prefix[1:] + text + "."
    if prefix:
        return text[0].upper() + text[1:] + "."
    if _ATOM_START.match(text):
        if at_start or rng.random() >= 0.22:
            return text + "."
        return rng.choice(_LEAD_INS) + text + "."
    return text[0].upper() + text[1:] + "."


def _plain_neg(lit: str) -> str:
    return f"not {lit[1:]}" if lit.startswith("-") else lit


def _lit_phrase(lit: str, rng, cyc=None) -> str:
    if lit.startswith("-"):
        return cyc.pick(NEGATED_LITERAL).format(p=lit[1:])
    return lit


def _rule_ref(o: Operation, rng, cyc=None) -> str:
    def plain(lit):
        return f"not {lit[1:]}" if lit.startswith("-") else lit
    tmpl = cyc.pick(RULE_REFERENCE) if cyc else rng.choice(RULE_REFERENCE)
    return tmpl.format(p=plain(o.antecedents[0]), q=plain(o.consequent))


class _Cycler:
    def __init__(self, rng):
        self.rng = rng
        self.pools = {}

    def pick(self, forms):
        key = id(forms)
        pool = self.pools.get(key)
        if not pool:
            pool = list(forms)
            self.rng.shuffle(pool)
            self.pools[key] = pool
        return pool.pop()


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Union[FItem, Rejected]:
    rng = random.Random(stable_seed(seed, level, ordering, "fm"))
    cyc = _Cycler(rng)
    lo = 3 + int((level - 1) * (40 - 3) / 14)
    hi = 5 + int((level - 1) * (50 - 5) / 14)
    target = rng.randint(lo, hi)

    n_units = max(1, round(target / 3.0)) if level <= 2 else max(2, round(target / 3.0))
    names = _names(stable_seed(seed, level, ordering, "nm"), 12 + n_units * 10)
    it = iter(names)
    ops: List[Operation] = []
    sentences: List[str] = []
    queried: List[str] = []
    ridx = [0]
    typed: List[Tuple[str, object]] = []
    n_neg = [0]
    j_budget = junction_budget(level, JUNCTION_CAPS["formalization"])
    j_used = [0]
    prof = PROFILES[profile]

    sentences.append(cyc.pick(OPENERS))
    for u in range(n_units):
        a, b, c = next(it), next(it), next(it)
        if j_used[0] < j_budget:
            j1, j2, jr = next(it), next(it), next(it)
            use_strict_j = (u % 8 == 5) and prof.permits("strict")
            ops.append(Operation(kind="premise", content=j1))
            ops.append(Operation(kind="premise", content=j2))
            sentences.append(_sentence(cyc.pick(PREMISE).format(p=j1), rng, at_start=True))
            sentences.append(_sentence(cyc.pick(PREMISE).format(p=j2), rng))
            ridx[0] += 1
            ops.append(Operation(kind="strict" if use_strict_j else "defeasible",
                                 name=f"q_{ridx[0]}", antecedents=(j1, j2), consequent=jr))
            sentences.append(_sentence(
                cyc.pick(JUNCTION_STRICT if use_strict_j else JUNCTION_DEFEASIBLE).format(
                    p=j1, q=j2, r=_lit_phrase(jr, rng, cyc)), rng,
                prefix=cyc.pick(FORWARD_CONNECTIVES)))
            typed.append(("strict" if use_strict_j else "defeasible", ops[-1]))
            if jr not in queried:
                queried.append(jr)
            j_used[0] += 1
            continue

        neg_unit = level >= 3 and u % 3 == 2
        if neg_unit:
            _avail = [1, 2]
            if prof.permits("axiom"):
                _avail.append(0)
            if prof.permits("strict"):
                _avail.append(3)
            if prof.permits("axiom") and prof.permits("strict"):
                _avail.append(4)
            _avail.sort()
            kind = _avail[n_neg[0] % len(_avail)]
            n_neg[0] += 1
            if kind == 0:
                ops.append(Operation(kind="axiom", content="-" + a))
                sentences.append(_sentence(cyc.pick(NEGATED_AXIOM).format(p=a), rng, at_start=True))
                ops.append(Operation(kind="premise", content=a))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=a), rng))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"q_{ridx[0]}",
                                     antecedents=("-" + a,), consequent=b))
                sentences.append(_sentence(
                    cyc.pick(DEFEASIBLE).format(p=_plain_neg("-" + a),
                                                q=_lit_phrase(b, rng, cyc)), rng,
                    prefix=cyc.pick(FORWARD_CONNECTIVES)))
            elif kind == 1:
                ops.append(Operation(kind="premise", content=a))
                ops.append(Operation(kind="premise", content="-" + a))
                ops.append(Operation(kind="prefer_premise", stronger="-" + a, weaker=a))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=a), rng, at_start=True))
                sentences.append(_sentence(cyc.pick(NEGATED_PREMISE).format(p=a), rng))
                sentences.append(_sentence(cyc.pick(PREFER_PREMISE).format(
                    p=_lit_phrase("-" + a, rng, cyc), q=a), rng))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"q_{ridx[0]}",
                                     antecedents=("-" + a,), consequent=b))
                sentences.append(_sentence(
                    cyc.pick(DEFEASIBLE).format(p=_plain_neg("-" + a),
                                                q=_lit_phrase(b, rng, cyc)), rng,
                    prefix=cyc.pick(FORWARD_CONNECTIVES)))
            elif kind == 2:
                ops.append(Operation(kind="premise", content=a))
                ops.append(Operation(kind="premise", content=c))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=a), rng, at_start=True))
                ridx[0] += 1
                pro = f"q_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=pro, antecedents=(a,), consequent=b))
                sentences.append(_sentence(
                    cyc.pick(DEFEASIBLE).format(p=a, q=_lit_phrase(b, rng, cyc)), rng,
                    prefix=cyc.pick(FORWARD_CONNECTIVES)))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=c), rng))
                ridx[0] += 1
                con = f"q_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=con, antecedents=(c,),
                                     consequent="-" + b))
                sentences.append(_sentence(cyc.pick(REBUT_RULE).format(p=c, q=b), rng,
                                           prefix=cyc.pick(OBJECTION_CONNECTIVES)))
                ops.append(Operation(kind="prefer_rule", stronger=pro, weaker=con))
                sentences.append(_sentence(cyc.pick(PREFER_RULE).format(
                    R1=_rule_ref(ops[-3], rng, cyc), R2=_rule_ref(ops[-2], rng, cyc)), rng))
            elif kind == 4:
                ops.append(Operation(kind="axiom", content="-" + a))
                sentences.append(_sentence(cyc.pick(NEGATED_AXIOM).format(p=a), rng, at_start=True))
                ridx[0] += 1
                ops.append(Operation(kind="strict", name=f"q_{ridx[0]}",
                                     antecedents=("-" + a,), consequent=b))
                sentences.append(_sentence(
                    cyc.pick(STRICT_FROM_NEGATION).format(p=a, q=_lit_phrase(b, rng, cyc)), rng,
                    prefix=cyc.pick(FORWARD_CONNECTIVES)))
                ops.append(Operation(kind="premise", content=c))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=c), rng))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"q_{ridx[0]}", antecedents=(c,),
                                     consequent="-" + b))
                sentences.append(_sentence(cyc.pick(REBUT_RULE).format(p=c, q=b), rng,
                                           prefix=cyc.pick(OBJECTION_CONNECTIVES)))
            else:
                ops.append(Operation(kind="premise", content=a))
                ops.append(Operation(kind="premise", content=c))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=a), rng, at_start=True))
                ridx[0] += 1
                ops.append(Operation(kind="strict", name=f"q_{ridx[0]}", antecedents=(a,),
                                     consequent="-" + b))
                sentences.append(_sentence(cyc.pick(STRICT_EXCLUSION).format(p=a, q=b), rng))
                sentences.append(_sentence(cyc.pick(PREMISE).format(p=c), rng))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"q_{ridx[0]}", antecedents=(c,),
                                     consequent=b))
                sentences.append(_sentence(
                    cyc.pick(DEFEASIBLE).format(p=c, q=_lit_phrase(b, rng, cyc)), rng,
                    prefix=cyc.pick(FORWARD_CONNECTIVES)))
            if b not in queried:
                queried.append(b)
            continue

        test_root = (u % 2 == 0)

        second_route = u > 0 and queried and rng.random() < 0.4
        # The draw stays where it is, so every item built before this keeps its
        # stream; only what the draw buys changes. A second route to the one claim
        # the item has queried makes every rule redundant to that claim, and an
        # item with no rule its queried statuses rest on cannot fail `success` on
        # an answer missing any rule -- three cells at levels 2 and 4 did that
        # (#121). With two or more claims queried, redundant support is the point.
        if second_route and len(queried) < 2:
            second_route = False
        if second_route:
            b = rng.choice(queried)
            sentences.append(cyc.pick(LINE_TRANSITIONS).format(q=b))
        elif u > 0:
            sentences.append(cyc.pick(NEW_TOPIC))

        if test_root:
            use_axiom = prof.permits("axiom") and rng.random() < 0.5
            root_kind = "axiom" if use_axiom else "premise"
            ops.append(Operation(kind=root_kind, content=a))
            sentences.append(_sentence(
                cyc.pick(AXIOM if use_axiom else PREMISE).format(p=a), rng, at_start=True))
            typed.append((root_kind, ops[-1]))
            ridx[0] += 1
            rname = f"q_{ridx[0]}"
            ops.append(Operation(kind="defeasible", name=rname, antecedents=(a,), consequent=b))
            sentences.append(_sentence(
                cyc.pick(DEFEASIBLE).format(p=a, q=_lit_phrase(b, rng, cyc)), rng,
                prefix=cyc.pick(FORWARD_CONNECTIVES)))
        else:
            use_strict = prof.permits("strict") and rng.random() < 0.5
            ops.append(Operation(kind="premise", content=a))
            sentences.append(_sentence(cyc.pick(PREMISE).format(p=a), rng, at_start=True))
            ridx[0] += 1
            rname = f"q_{ridx[0]}"
            rkind = "strict" if use_strict else "defeasible"
            rule_op = Operation(kind=rkind, name=rname, antecedents=(a,), consequent=b)
            ops.append(rule_op)
            sentences.append(_sentence(
                cyc.pick(STRICT if use_strict else DEFEASIBLE).format(
                    p=a, q=_lit_phrase(b, rng, cyc)), rng,
                prefix=cyc.pick(FORWARD_CONNECTIVES)))
            typed.append((rkind, rule_op))
            ops.append(Operation(kind="premise", content=c))
            ridx[0] += 1
            uname = f"q_{ridx[0]}"
            ops.append(Operation(kind="defeasible", name=uname, antecedents=(c,),
                                 consequent="-" + rname))
            sentences.append(_sentence(cyc.pick(PREMISE).format(p=c), rng))
            ref = (cyc.pick(RULE_ANAPHORA) if rng.random() < 0.55
                   else _rule_ref(rule_op, rng, cyc))
            sentences.append(_sentence(
                cyc.pick(UNDERCUT).format(r=c, R1=ref), rng,
                prefix=cyc.pick(OBJECTION_CONNECTIVES)))
        if b not in queried:
            queried.append(b)

    base = [o for o in ops if o.kind in ("premise", "axiom")] + \
           [o for o in ops if o.kind in ("defeasible", "strict")] + \
           [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]

    gold = status_map(base, queried, ordering)
    if not gold or len(gold) != len(queried):
        return Rejected("status_map_incomplete")

    flips = 0

    nl = " ".join(sentences)
    concl = "; ".join(f"{l} is {gold[l].lower()}" for l in queried)
    prompt = _render_prompt(nl, concl, ordering, queried)
    return FItem(
        prompt=prompt, nl_text=nl, reference_ops=base,
        reference="\n".join(render_op(o) for o in base),
        queried=queried, gold_status=gold, ordering=ordering, level=level,
        metadata={
            "n_directives": len(base), "n_units": n_units,
            "n_axioms": len([o for o in base if o.kind == "axiom"]),
            "n_strict": len([o for o in base if o.kind == "strict"]),
            "n_typed_exercised": flips, "n_negation_units": n_neg[0],
            "axiom_units_enabled": False,
            "axiom_discriminator_status": "unsolved -- see build() comment",
            "n_queried": len(queried),
            "target_range": (lo, hi),
        })


def _render_prompt(nl: str, concl: str, ordering: str, queried: Sequence[str]) -> str:
    on = _ordering_phrase(ordering)
    return (f"The following argumentation is described in words. Formalize it as an ASPIC+ theory, "
            f"to be evaluated under grounded semantics with {on}.\n\n{nl}\n\n"
            f"Under a correct formalization: {concl}.\n\n"
            + formalization_notation() + "\n\n"
            + answer_format("Answer format: one directive per line."))


def parse(text: str, item: Optional[FItem] = None) -> List[Operation]:
    """The theory an answer submits, as values. One DSL, one parser.

    A solver emitting operations directly hands them to `score_value` and never
    writes a directive (`docs/dataset-contract.md` section 4). `item` is accepted so
    every task parses through the same signature; formalization needs nothing from it.

    This module carried its own copy of the three directive patterns and the same
    unit-and-stray scan, and the copy drifted. It read an atom as `-?\\w+` where the
    scorer reads `-?[A-Za-z]\\w*`, so `[premise: 9x]` and `[defeasible r1: p => _x]`
    parsed here and were unparseable lines in every construction task; it required
    `premise:` with no space where the scorer allows one; and it kept an empty
    antecedent that the scorer rejects. Generated atoms match `^[a-z]{2}\\d`, so the
    stricter class is the one every theory is written in, and a second definition of
    the DSL is how the two fell out of step in the first place.
    """
    p = parse_answer(text)
    if p.n_unparseable:
        raise UnparseableAnswer(f"unparseable_tokens:{p.n_unparseable}",
                                {"n_parsed": len(p.ops),
                                 "n_unparseable": p.n_unparseable})
    return p.ops


def score(answer_text: str, item: FItem) -> ScoreResult:
    """`score_value(parse(text, item), item)`, with a parse failure turned into a row."""
    try:
        ops = parse(answer_text, item)
    except UnparseableAnswer as e:
        diag: Dict = {"n_parsed": 0, "n_unparseable": 0, "behavioural": None,
                      "gold_status": item.gold_status}
        diag.update(e.diagnostics)
        return ScoreResult(0.0, False, e.reason, diag)
    return score_value(ops, item)

def _alpha_shape_keys(ops: Sequence[Operation]):
    """Return structural keys invariant to model-chosen rule names.

    Rule names are local identifiers in formalization. Structural scoring
    therefore compares the rules they denote, including rule-name references
    in undercuts and rule preferences, rather than comparing identifier text.
    """
    rules = {
        o.name: o
        for o in ops
        if o.kind in ("defeasible", "strict") and o.name
    }
    cache = {}

    def rule_key(name: str, stack=()):
        if name not in rules:
            return ("unknown_rule", name)

        if name in cache:
            return cache[name]

        if name in stack:
            # Generated formalization theories should not contain recursive
            # rule-name references. Keep malformed submitted structures finite.
            return ("rule_ref_cycle",)

        o = rules[name]
        nxt = stack + (name,)

        k = (
            o.kind,
            tuple(
                sorted(
                    literal_key(a, nxt)
                    for a in (o.antecedents or ())
                )
            ),
            literal_key(o.consequent, nxt),
        )
        cache[name] = k
        return k

    def literal_key(lit, stack=()):
        if lit is None:
            return ("literal", None)

        text = str(lit)
        negated = text.startswith("-")
        bare = text[1:] if negated else text

        if bare in rules:
            return (
                "rule_ref",
                negated,
                rule_key(bare, stack),
            )

        return ("literal", text)

    def op_key(o: Operation):
        if o.kind in ("premise", "axiom"):
            return (
                o.kind,
                ("literal", o.content),
            )

        if o.kind in ("defeasible", "strict"):
            return rule_key(o.name)

        if o.kind == "prefer_rule":
            return (
                o.kind,
                rule_key(o.stronger),
                rule_key(o.weaker),
            )

        if o.kind == "prefer_premise":
            return (
                o.kind,
                ("literal", o.stronger),
                ("literal", o.weaker),
            )

        return (o.kind,)

    return [op_key(o) for o in ops]

def score_value(ops: Sequence[Operation], item: FItem) -> ScoreResult:
    ops = list(ops)
    diag: Dict = {"n_parsed": len(ops), "n_unparseable": 0, "behavioural": None,
                  "gold_status": item.gold_status}
    if not ops:
        return ScoreResult(0.0, False, "no_directives", diag)

    ordered = [o for o in ops if o.kind in ("premise", "axiom")] + \
              [o for o in ops if o.kind in ("defeasible", "strict")] + \
              [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
    got = status_map(ordered, item.queried, item.ordering)
    if not got:
        return ScoreResult(0.0, False, "engine_rejected", diag)
    gold_pairs = {(l, item.gold_status[l]) for l in item.queried}
    pred_pairs = {(l, got[l]) for l in item.queried if l in got and got[l] != "UNSATISFIABLE"}
    tp = len(gold_pairs & pred_pairs)
    b_prec = tp / max(len(pred_pairs), 1)
    b_rec = tp / max(len(gold_pairs), 1)
    behavioural = 0.0 if b_prec + b_rec == 0 else 2 * b_prec * b_rec / (b_prec + b_rec)
    diag["behavioural"] = round(behavioural, 4)
    diag["behavioural_precision"] = round(b_prec, 4)
    diag["behavioural_recall"] = round(b_rec, 4)
    diag["got_status"] = got
    gold_shape_keys = _alpha_shape_keys(item.reference_ops)
    pred_shape_keys = _alpha_shape_keys(ordered)
    gset = collections.Counter(gold_shape_keys)
    pset = collections.Counter(pred_shape_keys)
    inter = sum(min(gset[k], pset[k]) for k in set(gset) | set(pset))
    prec = inter / max(sum(pset.values()), 1)
    rec = inter / max(sum(gset.values()), 1)
    shape = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    diag["shape_f1"] = round(shape, 4)
    diag["missed_directives"] = [str(k) for k in (gset - pset)][:5]
    diag["extra_directives"] = [str(k) for k in (pset - gset)][:5]

    contested_keys = [
        k
        for o, k in zip(item.reference_ops, gold_shape_keys)
        if o.kind in ("axiom", "strict")
    ]

    if contested_keys:
        typed_gold = collections.Counter(contested_keys)

        got_types = sum(
            min(n, pset[k])
            for k, n in typed_gold.items()
        )

        type_score = got_types / len(contested_keys)

        diag["type_decisions_total"] = len(contested_keys)
        diag["type_decisions_correct"] = got_types
    else:
        type_score = None

    if type_score is None:
        total = round(0.4 * behavioural + 0.6 * shape, 4)
    else:
        total = round(0.25 * behavioural + 0.35 * shape + 0.40 * type_score, 4)
    # Success is behavioural equivalence alone. The prompt states its own success
    # condition behaviourally ("Under a correct formalization: ..."), so a theory that
    # reproduces every queried status is what was asked for, whatever names and groupings
    # it used; requiring shape_f1 == 1.0 would demand the reference's exact directive
    # multiset (docs/dataset-contract.md, section 5).
    exact_behaviour = behavioural >= 0.999
    diag.update(directives_f1=round(shape, 4), directives_correct=inter,
                directives_gold=sum(gset.values()), directives_written=sum(pset.values()),
                type_score=None if type_score is None else round(type_score, 4),
                shape_f1=round(shape, 4), exact_behaviour=exact_behaviour)
    return ScoreResult(total, exact_behaviour, "ok", diag)


def make_item_report(level: int, seed: int, ordering: str = LAST_LINK,
                     profile: str = "FULL", tries: int = 16) -> BuildReport:
    return retry(lambda k: build(level, seed * 89 + k, ordering, profile=profile), tries)


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              profile: str = "FULL",
              tries: int = 16) -> Optional[FItem]:
    return make_item_report(level, seed, ordering, profile, tries).item
