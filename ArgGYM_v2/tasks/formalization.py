from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier
from core.curriculum import junction_budget, JUNCTION_CAPS, PROFILES
from core.nlforms import (AXIOM, DEFEASIBLE, FORWARD_CONNECTIVES, LINE_TRANSITIONS,
                          NEGATED_AXIOM, NEGATED_LITERAL, NEGATED_PREMISE, REBUT_RULE,
                          STRICT_EXCLUSION, STRICT_FROM_NEGATION,
                          JUNCTION_DEFEASIBLE, JUNCTION_STRICT,
                     NEW_TOPIC, OBJECTION_CONNECTIVES, OPENERS, PREFER_PREMISE, PREFER_RULE,
                     PREMISE, RULE_ANAPHORA, RULE_REFERENCE, STRICT, UNDERCUT)

TASK = "formalization"
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
          profile: str = "FULL") -> Optional[FItem]:
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
        return None

    flips = 0

    nl = " ".join(sentences)
    concl = "; ".join(f"{l} is {gold[l].lower()}" for l in queried)
    prompt = _render_prompt(nl, concl, ordering, queried)
    return FItem(
        prompt=prompt, nl_text=nl, reference_ops=base,
        reference="[answer]\n" + "\n".join(render_op(o) for o in base) + "\n[/answer]",
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
    on = "the last-link strength ordering" if ordering == LAST_LINK \
        else "the weakest-link strength ordering"
    return (f"The following argumentation is described in words. Formalize it as an ASPIC+ theory, "
            f"to be evaluated under grounded semantics with {on}.\n\n{nl}\n\n"
            f"Under a correct formalization: {concl}.\n\n"
            "Notation: [premise: x], [axiom: x], [defeasible name: a => b], [strict name: a -> b], "
            "[prefer_rule: r1 > r2], [prefer_premise: x > y]. Negation is written -x. "
            "Rule names are yours to choose.\n\n"
            "Answer format: one directive per line, between [answer] and [/answer].")


# Answer region comes from core.scoring: the LAST complete region, so a
# reasoning model that drafts and then revises is scored on the revision.
from core.scoring import answer_region
_P = re.compile(r"^\[(premise|axiom):\s*(-?\w+)\]$")
_R = re.compile(r"^\[(defeasible|strict)\s+([\w]+)\s*:\s*(.+?)\s*(=>|->)\s*(-?\w+)\]$")
_F = re.compile(r"^\[prefer_(rule|premise):\s*(-?\w+)\s*>\s*(-?\w+)\]$")


def parse(text: str) -> Tuple[List[Operation], int]:
    m = answer_region(text)
    if m is None:
        return [], -1
    ops, bad = [], 0
    body = m
    units = re.findall(r"\[[^\]]*\]", body)
    leftover = re.sub(r"\[[^\]]*\]", " ", body)
    stray = [t for t in leftover.split()
             if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    for raw in units + stray:
        line = raw.strip()
        if not line:
            continue
        mm = _P.match(line)
        if mm:
            ops.append(Operation(kind=mm.group(1), content=mm.group(2)))
            continue
        mm = _R.match(line)
        if mm:
            ops.append(Operation(kind=mm.group(1), name=mm.group(2),
                                 antecedents=tuple(a.strip() for a in mm.group(3).split("AND")),
                                 consequent=mm.group(5)))
            continue
        mm = _F.match(line)
        if mm:
            ops.append(Operation(kind="prefer_" + mm.group(1), stronger=mm.group(2),
                                 weaker=mm.group(3)))
            continue
        bad += 1
    return ops, bad


def score(answer_text: str, item: FItem, strict_parse: bool = True) -> Dict:
    diag: Dict = {"n_parsed": 0, "n_unparseable": 0, "behavioural": None,
                  "gold_status": item.gold_status}
    ops, bad = parse(answer_text)
    if bad < 0:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    diag["n_parsed"], diag["n_unparseable"] = len(ops), bad
    if strict_parse and bad:
        return {"score": 0.0, "reason": f"unparseable_lines:{bad}", "diagnostics": diag}
    if not ops:
        return {"score": 0.0, "reason": "no_directives", "diagnostics": diag}

    ordered = [o for o in ops if o.kind in ("premise", "axiom")] + \
              [o for o in ops if o.kind in ("defeasible", "strict")] + \
              [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
    got = status_map(ordered, item.queried, item.ordering)
    if not got:
        return {"score": 0.0, "reason": "engine_rejected", "diagnostics": diag}
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

    def key(o):
        if o.kind in ("premise", "axiom"):
            return (o.kind, o.content)
        if o.kind in ("defeasible", "strict"):
            return (o.kind, tuple(sorted(o.antecedents or ())), o.consequent)
        return (o.kind,)
    import collections
    gset = collections.Counter(key(o) for o in item.reference_ops)
    pset = collections.Counter(key(o) for o in ordered)
    inter = sum(min(gset[k], pset[k]) for k in set(gset) | set(pset))
    prec = inter / max(sum(pset.values()), 1)
    rec = inter / max(sum(gset.values()), 1)
    shape = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    diag["shape_f1"] = round(shape, 4)
    diag["missed_directives"] = [str(k) for k in (gset - pset)][:5]
    diag["extra_directives"] = [str(k) for k in (pset - gset)][:5]

    contested = [o for o in item.reference_ops if o.kind in ("axiom", "strict")]
    if contested:
        pk_keys = collections.Counter(key(o) for o in ordered)
        got_types = sum(1 for o in contested if pk_keys[key(o)] > 0)
        type_score = got_types / len(contested)
        diag["type_decisions_total"] = len(contested)
        diag["type_decisions_correct"] = got_types
    else:
        type_score = None

    if type_score is None:
        total = round(0.4 * behavioural + 0.6 * shape, 4)
    else:
        total = round(0.25 * behavioural + 0.35 * shape + 0.40 * type_score, 4)
    return {"score": total, "reason": "ok",
            "directives_f1": round(shape, 4),
            "directives_correct": inter, "directives_gold": sum(gset.values()),
            "directives_written": sum(pset.values()),
            "type_score": None if type_score is None else round(type_score, 4),
            "behavioural": round(behavioural, 4), "shape_f1": round(shape, 4),
            "exact_behaviour": behavioural >= 0.999, "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              profile: str = "FULL",
              tries: int = 16) -> Optional[FItem]:
    for k in range(tries):
        it = build(level, seed * 89 + k, ordering, profile=profile)
        if it is not None:
            return it
    return None
