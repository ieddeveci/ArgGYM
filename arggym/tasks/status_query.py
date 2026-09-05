from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.core.answers import AnswerTemplate, ScoreResult, UnparseableAnswer
from arggym.core.curriculum import PROFILES, junctions_for
from arggym.core.invariants import randomize_rule_names, split_atoms_and_rules
from arggym.core.pairs import collect, pair_f1
from arggym.core.prompting import answer_format

TASK = "status_query"
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


def _weakest_link_split(ops, names, ridx, want_split: bool):
    from arggym.aspic.engine import Operation
    pa, pb, tbase, la, lb, goal = (next(names) for _ in range(6))
    ridx[0] += 1
    ja = f"q_{ridx[0]}"
    ridx[0] += 1
    jb = f"q_{ridx[0]}"
    ridx[0] += 1
    join = f"q_{ridx[0]}"
    ridx[0] += 1
    sup = f"q_{ridx[0]}"
    ops.extend([
        Operation(kind="premise", content=pa),
        Operation(kind="premise", content=pb),
        Operation(kind="premise", content=tbase),
        Operation(kind="defeasible", name=ja, antecedents=(pa,), consequent=la),
        Operation(kind="defeasible", name=jb, antecedents=(pb,), consequent=lb),
        Operation(kind="defeasible", name=join, antecedents=(la, lb), consequent="-" + goal),
        Operation(kind="defeasible", name=sup, antecedents=(tbase,), consequent=goal),
        Operation(kind="prefer_premise", stronger=tbase, weaker=pa),
        # Every rule of the join argument, not just the first. Weakest-link reads the
        # rule set as well as the premise set, so with `jb` and `join` left unranked the
        # two arguments stayed incomparable on the rules whatever the premises said, and
        # the `want_split` premise below could not change any status.
        Operation(kind="prefer_rule", stronger=sup, weaker=ja),
        Operation(kind="prefer_rule", stronger=sup, weaker=jb),
        Operation(kind="prefer_rule", stronger=sup, weaker=join),
    ])
    # The discriminator. Ranked, `{pa, pb}` is weaker than `{tbase}` under both readings.
    # Left split, only `pa` is weaker: elitist needs one such element and defeats the join
    # argument, democratic needs every element and does not.
    if not want_split:
        ops.append(Operation(kind="prefer_premise", stronger=tbase, weaker=pb))
    return goal


def _several_last_links(ops, names, ridx, want_split: bool):
    from arggym.aspic.engine import Operation
    leg1, leg2, base, part1, part2, thesis = (next(names) for _ in range(6))
    ridx[0] += 1
    arm1 = f"v_{ridx[0]}"
    ridx[0] += 1
    arm2 = f"v_{ridx[0]}"
    ridx[0] += 1
    joiner = f"v_{ridx[0]}"
    ridx[0] += 1
    support = f"v_{ridx[0]}"
    ops.extend([
        Operation(kind="premise", content=leg1),
        Operation(kind="premise", content=leg2),
        Operation(kind="premise", content=base),
        Operation(kind="defeasible", name=arm1, antecedents=(leg1,), consequent=part1),
        Operation(kind="defeasible", name=arm2, antecedents=(leg2,), consequent=part2),
        Operation(kind="strict", name=joiner, antecedents=(part1, part2),
                  consequent="-" + thesis),
        Operation(kind="defeasible", name=support, antecedents=(base,), consequent=thesis),
        Operation(kind="prefer_rule", stronger=support, weaker=arm1),
    ])
    if not want_split:
        ops.append(Operation(kind="prefer_rule", stronger=support, weaker=arm2))
    return thesis


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
          profile: str = "FULL",
          template: Optional[AnswerTemplate] = None) -> Optional[SQItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "sq", profile))
    prof = PROFILES[profile]
    n_query = max(3, round(3 + (level - 1) * (40 - 3) / 14))
    n_group = max(3, round(n_query / 1.6))
    max_tower = 0 if level < 5 else min(1 + (level - 5) // 4, 3)
    j_budget = junctions_for(level, max(1, n_group * 3))
    # The junction takes the whole group, so every group it lands on skips the `want`
    # dispatch below and the two blocks after it. Uncapped it exceeded the number of
    # groups it is allowed to land on, which left three shapes off the grid entirely:
    # justified-with-tower, overruled-by-undermined-premise, and overruled-by-rebuttal.
    # Half the eligible groups keep the junction and half keep their own shape.
    _eligible = sum(1 for g in range(n_group) if STATUSES[g % 3] != "UNDECIDED")
    j_budget = min(j_budget, max(1, _eligible // 2))
    j_used = [0]
    shapes: collections.Counter = collections.Counter()

    names = _names(stable_seed(seed, (level, ordering, "nm") * 3), 40 + n_group * 12)
    it = iter(names)
    ops: List[Operation] = []
    ridx = [0]
    planned: List[Tuple[str, str]] = []

    for g in range(n_group):
        want = STATUSES[g % 3]
        root, mid = next(it), next(it)

        # The junction plants one OVERRULED and one JUSTIFIED literal and takes the whole
        # group, so a group it lands on never reaches the `want` dispatch below. The budget
        # is spent on a prefix of the groups, which meant it ate the undecided third from
        # the front: at level 12, six groups ask for UNDECIDED and one used to survive.
        # Undecided literals are the scarce status, and the 0.45 share cap turns a shortage
        # of them into a shortage of queries, so the junction gives way to them here.
        if (want != "UNDECIDED"
                and j_used[0] < j_budget
                and prof.permits("defeasible")):
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
            shapes["junction"] += 1
            continue

        if not prof.permits("defeasible"):
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
            shapes["no_defeasible"] += 1
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
        shapes[want.lower()] += 1
        planned.append((mid, want))

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

    _last_split = _weakest_split = None
    if prof.permits("strict") and prof.permits("prefer_rule") and level >= 4:
        _last_split = level % 2 == 0
        _t = _several_last_links(ops, it, ridx, want_split=_last_split)
        planned.append((_t, None))
    if prof.permits("prefer_premise") and prof.permits("defeasible") and level >= 4:
        # Drawn rather than derived from the level. The exported grid steps by three, so a
        # remainder taken against three is the same number at every level it is asked
        # about, and the split branch reached no shipped item (#69).
        _weakest_split = stable_seed(seed, level, ordering, "wls") % 2 == 0
        _w = _weakest_link_split(ops, it, ridx, want_split=_weakest_split)
        planned.append((_w, None))

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

    # A short item was shipped rather than rejected, so `n_queried == target_n_query` held
    # by luck of the rng stream and nothing put a build back for being under length. That
    # is what left two level-3 cells at 8 of 8 short: `make_item` already walks 24 build
    # seeds and only retries when `build` returns None.
    if len(queried) < n_query:
        return None

    gold = {l: sm[l] for l in queried}
    counts = collections.Counter(gold.values())
    if len(counts) < 3:
        return None
    if max(counts.values()) / len(gold) > MAX_STATUS_SHARE:
        return None

    lines = [f"{l}: {gold[l].lower()}" for l in queried]
    prompt = _render_prompt(render_ops(base), queried, ordering, template)
    return SQItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, queried=queried, gold=gold,
        ordering=ordering, level=level,
        reference="\n".join(lines),
        metadata={
            "n_queried": len(queried), "n_groups": n_group, "max_tower": max_tower,
            "n_items": len(base), "target_n_query": n_query,
            "status_counts": dict(counts),
            "modal_share": round(max(counts.values()) / len(gold), 4),
            "profile": profile,
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
            "n_axioms": len([o for o in base if o.kind == "axiom"]),
            "n_strict": len([o for o in base if o.kind == "strict"]),
            "group_shapes": dict(shapes),
            # None where the level is below the guard and the construct is not built.
            "last_link_split": _last_split, "weakest_link_split": _weakest_split,
        })


def _render_prompt(theory: str, queried: Sequence[str], ordering: str,
                   template: Optional[AnswerTemplate] = None) -> str:
    on = _ordering_phrase(ordering)
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"State the status of each of the following claims: {', '.join(queried)}.\n"
            "Possible statuses: justified, overruled, undecided.\n"
            "A claim is justified when some argument for it is accepted, overruled when every argument for it is defeated, and undecided otherwise.\n"
            "\n"
            + answer_format("Answer format: one line per claim, written as `claim: status`.",
                            template))


# One whole line, as the format clause above asks for. See `core/pairs.py` for why the
# anchors are there: unanchored, this read "My conclusion: overruled" as a prediction
# about a claim called `conclusion`.
_PAIR = re.compile(r"(-?\w+)\s*[:=]\s*(justified|overruled|undecided)\b(?!\w)", re.I)


#: A claim keyed to the status the answer gives it. A sequence where the answer gave
#: several statuses for one claim: only text can contradict itself, so a solver that
#: submits a mapping writes one status per claim and a text answer that hedges is the
#: only way to reach the `contradicted` diagnostic (`core/pairs.py:pair_f1`).
Value = Dict[str, Union[str, Sequence[str]]]


def _diagnostics(item: SQItem) -> Dict:
    """The keys every branch returns, so a harness never tests for one before reading it."""
    return {"n_predicted": 0, "n_gold": len(item.gold), "wrong": [], "missing": [],
            "contradicted": [], "n_unparseable": 0, "junk_tokens": []}


def parse(answer_text: str, item: SQItem) -> Value:
    """The claim-status pairs the answer states, in the order it states them.

    A status repeated for one claim is kept rather than folded away here: whether saying
    the same thing twice is one prediction is a scoring question, and `score_value`
    answers it.
    """
    body = answer_text or ""
    said: Dict[str, List[str]] = {}
    for claim, stat in _PAIR.findall(body):
        said.setdefault(claim.lower(), []).append(stat.upper())
    residue = _PAIR.sub(" ", body)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    if junk:
        raise UnparseableAnswer(f"unparseable_tokens:{len(junk)}",
                                {"n_unparseable": len(junk), "junk_tokens": junk[:6]})
    if not said:
        raise UnparseableAnswer("no_parseable_lines")
    return said


def score_value(value: Value, item: SQItem) -> ScoreResult:
    """Grade a label map, whichever way it arrived.

    A solver with constrained decoding, a JSON schema or a tool call submits one of these
    and never imitates the line format (`docs/dataset-contract.md` section 4).
    """
    diag = _diagnostics(item)
    pred = collect((claim.lower(), stat.upper())
                   for claim, statuses in value.items()
                   for stat in ((statuses,) if isinstance(statuses, str) else statuses))
    diag["n_predicted"] = len(pred)

    r = pair_f1(pred, item.gold)
    diag["wrong"] = sorted(f"{k}:said {'/'.join(v)}, is {item.gold[k]}"
                           for k, v in pred.items() if k in item.gold and v != [item.gold[k]])[:6]
    diag["missing"] = sorted(k for k in item.gold if k not in pred)[:6]
    diag["contradicted"] = r.contradicted[:6]
    diag["n_correct"] = r.tp
    diag.update(f1=round(r.f1, 4), precision=round(r.precision, 4),
                recall=round(r.recall, 4), exact_match=r.exact_match)
    return ScoreResult(round(r.f1, 4), r.exact_match, "ok", diag)


def score(answer_text: str, item: SQItem) -> ScoreResult:
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
              profile: str = "FULL", tries: int = 24,
              template: Optional[AnswerTemplate] = None) -> Optional[SQItem]:
    for k in range(tries):
        it = build(level, seed * 97 + k, ordering, profile, template)
        if it is not None:
            return it
    return None
