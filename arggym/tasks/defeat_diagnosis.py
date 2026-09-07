from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.core.answers import ScoreResult, UnparseableAnswer
from arggym.core.build import BuildReport, Rejected, retry
from arggym.core.curriculum import (
    JUNCTION_CAPS,
    PROFILES,
    junction_budget,
    junctions_for,
    negated_branch,
    wants_ternary,
)
from arggym.core.invariants import randomize_rule_names, split_atoms_and_rules
from arggym.core.prompting import answer_format

TASK = "defeat_diagnosis"
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

UNDERMINE, UNDERCUT, REBUT = "undermine", "undercut", "rebut"


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
class DDItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    claim: str
    claim_status: str
    diagnoses: List[Dict]
    ordering: str
    level: int
    reference: str
    metadata: Dict = field(default_factory=dict)


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Union[DDItem, Rejected]:
    rng = random.Random(stable_seed(seed, level, ordering, "dd"))
    n_routes = 1 if level <= EASY_LEVELS else 3
    depth = max(2, min(2 + level // 3, 7))
    tower = 0 if level < 5 else min(1 + (level - 5) // 4, 3)
    use_junction = level >= 8
    n_filler = max(0, min(2 + level * 2, 30))
    n_inert = 0 if level < 4 else min(1 + (level - 4) // 5, 3)

    names = _names(stable_seed(seed, level, ordering, "nm"),
                   40 + n_routes * (depth + 12) + n_filler * 8 + tower * 4)
    it = iter(names)
    claim = next(it)
    ops: List[Operation] = []
    ridx = [0]
    diagnoses: List[Dict] = []

    kinds = [UNDERMINE, UNDERCUT, REBUT]
    rng.shuffle(kinds)
    kind_seq = kinds[:n_routes]

    j_budget = min(3, junction_budget(level, JUNCTION_CAPS["defeat_diagnosis"]))
    j_routes = set(range(min(j_budget, n_routes))) if use_junction else set()
    for k in range(n_routes):
        kind = kind_seq[k]
        use_axiom = (level >= 4 and kind != UNDERMINE and k % 2 == 0
                     and PROFILES[profile].permits("axiom"))
        strict_at = ((depth // 2 + 1) if (level >= 6 and depth >= 3
                                        and PROFILES[profile].permits("strict")) else -1)
        if kind == UNDERCUT and strict_at == depth // 2:
            strict_at = -1
        use_neg_root = (level >= 5 and not use_axiom and k % 2 == 1
                        and kind != UNDERMINE)
        root = next(it)
        if use_neg_root:
            ops.append(Operation(kind="premise", content=root))
            ops.append(Operation(kind="premise", content="-" + root))
            ops.append(Operation(kind="prefer_premise", stronger="-" + root, weaker=root))
            root = "-" + root
        else:
            ops.append(Operation(kind="axiom" if use_axiom else "premise", content=root))
        cur = root
        rules: List[str] = []
        lits: List[str] = []
        j_here = set()
        if depth >= 3 and k in j_routes:
            j_here.add(depth // 2)
            if j_budget > n_routes and depth >= 5:
                j_here.add(max(1, depth // 4))
        for j in range(depth):
            ridx[0] += 1
            nm = f"r_{ridx[0]}"
            nxt = claim if j == depth - 1 else next(it)
            is_strict = (j == strict_at and j != depth - 1)
            if j in j_here and not is_strict:
                n_extra = 2 if wants_ternary(level, k) else 1
                extra = []
                for _e in range(n_extra):
                    broot = next(it)
                    _bsrc = ("-" + broot) if negated_branch(k) else broot
                    ops.append(Operation(kind="premise", content=_bsrc))
                    ridx[0] += 1
                    bnm = f"r_{ridx[0]}"
                    blit = next(it)
                    ops.append(Operation(kind="defeasible", name=bnm, antecedents=(_bsrc,),
                                         consequent=blit))
                    rules.append(bnm)
                    lits.append(blit)
                    extra.append(blit)
                ridx[0] += 1
                nm = f"r_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=nm,
                                     antecedents=tuple([cur] + extra), consequent=nxt))
            else:
                ops.append(Operation(kind="strict" if is_strict else "defeasible", name=nm,
                                     antecedents=(cur,), consequent=nxt))
            rules.append(nm)
            lits.append(nxt)
            cur = nxt

        src = next(it)
        ops.append(Operation(kind="premise", content=src))
        survives_because = None

        if kind == UNDERMINE:
            ops.append(Operation(kind="premise", content="-" + root))
            ops.append(Operation(kind="prefer_premise", stronger="-" + root, weaker=root))
            defeated_at, defeater = root, "-" + root
        elif kind == UNDERCUT:
            tgt_rule = rules[len(rules) // 2]
            ridx[0] += 1
            defeater = f"w_{ridx[0]}"
            ops.append(Operation(kind="defeasible", name=defeater, antecedents=(src,),
                                 consequent="-" + tgt_rule))
            defeated_at = tgt_rule
        else:
            tgt_lit = lits[len(lits) // 2]
            ridx[0] += 1
            defeater = f"w_{ridx[0]}"
            ops.append(Operation(kind="defeasible", name=defeater, antecedents=(src,),
                                 consequent="-" + tgt_lit))
            ops.append(Operation(kind="prefer_rule", stronger=defeater,
                                 weaker=rules[len(lits) // 2]))
            defeated_at = tgt_lit

        if tower and kind != UNDERMINE:
            prev = defeater
            chain = []
            for i in range(2 * tower):
                q = next(it)
                ops.append(Operation(kind="premise", content=q))
                ridx[0] += 1
                nm = f"w_{ridx[0]}"
                ops.append(Operation(kind="defeasible", name=nm, antecedents=(q,),
                                     consequent="-" + prev))
                chain.append(nm)
                prev = nm
            survives_because = chain[0] if chain else None

        diagnoses.append({"route": k, "kind": kind, "defeated_at": defeated_at,
                          "defeater": defeater, "survives_because": survives_because,
                          "axiom_rooted": use_axiom, "has_strict": strict_at >= 0,
                          "negated_root": use_neg_root})

    prof = PROFILES[profile]
    for _ in range(n_inert):
        ax = next(it)
        if prof.permits("axiom"):
            ops.append(Operation(kind="axiom", content=ax))
        else:
            ops.append(Operation(kind="premise", content=ax))
            ops.append(Operation(kind="premise", content="-" + ax))
            ops.append(Operation(kind="prefer_premise", stronger=ax, weaker="-" + ax))
        mid = next(it)
        ridx[0] += 1
        rmid = f"r_{ridx[0]}"
        ops.append(Operation(kind="defeasible", name=rmid, antecedents=(ax,), consequent=mid))
        ridx[0] += 1
        rstrict = f"r_{ridx[0]}"
        ops.append(Operation(kind="strict" if prof.permits("strict") else "defeasible",
                             name=rstrict, antecedents=(mid,), consequent=next(it)))
        q = next(it)
        ops.append(Operation(kind="premise", content=q))
        ridx[0] += 1
        ops.append(Operation(kind="defeasible", name=f"w_{ridx[0]}", antecedents=(q,),
                             consequent="-" + rstrict))

    _base = sum(1 for o in ops if o.kind == "defeasible")
    _have = sum(1 for o in ops if o.kind == "defeasible" and len(o.antecedents or ()) > 1)
    _fill_j = n_filler
    for _try in range(0, n_filler + 1):
        if (_have + _try) >= junctions_for(level, _base + n_filler + _try * 2, solve=False):
            _fill_j = _try
            break
    for _fi in range(n_filler):
        a, b = next(it), next(it)
        ops.append(Operation(kind="premise", content=a))
        ridx[0] += 1
        if _fi < _fill_j:
            _ex = []
            for _e in range(2 if (level >= 9 and _fi % 2 == 0) else 1):
                br, bl = next(it), next(it)
                _bsrc = ("-" + br) if negated_branch(_fi * 2 + _e) else br
                ops.append(Operation(kind="premise", content=_bsrc))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                     antecedents=(_bsrc,), consequent=bl))
                _ex.append(bl)
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                 antecedents=tuple([a] + _ex), consequent=b))
        else:
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(a,),
                                 consequent=b))

    ops, rmap = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    for d in diagnoses:
        d["defeater"] = rmap.get(d["defeater"], d["defeater"])
        d["defeated_at"] = rmap.get(d["defeated_at"], d["defeated_at"])
        if d["survives_because"]:
            d["survives_because"] = rmap.get(d["survives_because"], d["survives_because"])

    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))
    atoms, rnames = split_atoms_and_rules(base)
    if atoms & rnames:
        return Rejected("atom_rule_name_collision")

    st = status(base, claim, ordering)
    if st not in ("OVERRULED", "UNDECIDED"):
        return Rejected("claim_not_defeated")

    for d in diagnoses:
        sb = d.get("survives_because")
        if not sb:
            continue
        pruned = [o for o in base
                  if not (o.kind in ("defeasible", "strict") and o.consequent == "-" + sb)]
        if len(pruned) == len(base):
            d["survives_because"] = None
            continue
        if status(pruned, claim, ordering) == st:
            return Rejected("survival_reason_not_load_bearing")

    lines = [f"status: {st.lower()}"]
    for d in diagnoses:
        parts = [f"defeated_at: {d['defeated_at']}", f"defeater: {d['defeater']}",
                 f"kind: {d['kind']}"]
        if d["survives_because"]:
            parts.append(f"survives_because: {d['survives_because']}")
        lines.append("; ".join(parts))

    prompt = _render_prompt(render_ops(base), claim, ordering, bool(tower))
    return DDItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, claim=claim,
        claim_status=st, diagnoses=diagnoses, ordering=ordering, level=level,
        reference="\n".join(lines),
        metadata={
            "n_routes": n_routes, "chain_depth": depth, "tower": tower,
            "n_inert_decoys": n_inert,
            "axiom_rooted_routes": sum(1 for d in diagnoses if d.get("axiom_rooted")),
            "routes_with_strict": sum(1 for d in diagnoses if d.get("has_strict")),
            "negated_root_routes": sum(1 for d in diagnoses if d.get("negated_root")),
            "kinds": sorted({d["kind"] for d in diagnoses}),
            "requires_survival_reason": bool(tower),
            "n_items": len(base), "claim_status": st,
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
        })


def _render_prompt(theory: str, claim: str, ordering: str, want_survival: bool) -> str:
    on = _ordering_phrase(ordering)
    extra = "   ...; survives_because: <rule>\n" if want_survival else ""
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"The claim {claim} is not justified.\n"
            f"State its status, and identify every point at which its support fails.\n\n"
            + answer_format("Answer format:\n"
                            "   first line: `status: overruled` or `status: undecided`\n"
                            "   then one line per failure point, as\n"
                            "   `defeated_at: <target>; defeater: <defeater>; "
                            "kind: undermine|undercut|rebut`\n" + extra))


_STATUS = re.compile(r"status\s*[:=]\s*(justified|overruled|undecided)", re.I)
_STATUS_LINE = re.compile(r"^[\W\d]*status\s*[:=]\s*(justified|overruled|undecided)", re.I | re.M)
_FIELD = re.compile(r"(\w+)\s*[:=]\s*([^;\n]+)")


#: One failure point, carrying the fields the prompt names: `defeated_at`,
#: `defeater`, `kind`, and `survives_because` where the defeater is itself defeated.
DiagnosisRecord = Dict[str, str]
#: What a `defeat_diagnosis` answer is: the claim's status and its failure points.
#: `status` is a sequence because an answer may state more than one, and an answer
#: that states two is scored as the contradiction it is rather than as either of
#: them; a solver writing the value directly writes one.
DefeatDiagnosisAnswer = Dict[str, Any]

_KIND = re.compile(r"kind\s*[:=]\s*([^;\n,]+)", re.I)
_RECORD_START = re.compile(r"(?=defeated_at\s*[:=])")


def _blank_diagnostics(item: DDItem) -> Dict:
    """The keys every result carries, so a zero and a one have the same shape."""
    return {"n_quoted": 0, "n_gold": len(item.diagnoses),
            "status_correct": False, "status_contradicted": False,
            "kind_contradicted": False, "extra": [], "missing": []}


def parse(text: str, item: DDItem) -> DefeatDiagnosisAnswer:
    """The status an answer states and the failure points it lists, in its own order.

    The text arrives already extracted: composing the prompt and pulling the answer
    out of whatever came back is the harness's job, so the dataset never unwraps a
    fence. That is what lets a caller use any convention at all -- or none, with a
    solver that submits the value directly (`docs/dataset-contract.md` section 4).

    A record runs from its `defeated_at` to the next one, so `survives_because` on a
    line of its own belongs to the record above it. That reach stops at the next
    `defeated_at`: a record naming a target but no defeater is still a record, and a
    reason written under it is a reason about that failure point, not about the one
    before it.

    An empty answer parses to a status of nothing and no records. It is an answer
    with nothing in it, not an answer that failed to arrive.
    """
    body = text or ""
    for k in _KIND.findall(body):
        written = k.strip()
        if written.strip(",;.").lower() not in (UNDERMINE, UNDERCUT, REBUT):
            raise UnparseableAnswer(f"invalid_kind:{written[:16]}",
                                    {"invalid_kind": written})
    residue = _STATUS.sub(" ", body)
    residue = re.sub(r"(defeated_at|defeater|kind|survives_because)\s*[:=]\s*[^;\n]+",
                     " ", residue, flags=re.I)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    if junk:
        raise UnparseableAnswer(f"unparseable_tokens:{len(junk)}",
                                {"n_unparseable": len(junk), "junk_tokens": junk[:6]})

    records: List[DiagnosisRecord] = []
    # Everything before the first `defeated_at` is the status line and whatever led
    # up to it, so the records are the blocks after it.
    for block in _RECORD_START.split(body)[1:]:
        fields = {k.lower(): v.strip().strip(",;.") for k, v in _FIELD.findall(block)}
        if "defeated_at" not in fields:
            continue
        rec = {k: fields[k] for k in ("defeated_at", "defeater", "kind") if k in fields}
        if fields.get("survives_because"):
            rec["survives_because"] = fields["survives_because"]
        records.append(rec)
    return {"status": list(dict.fromkeys(_STATUS_LINE.findall(body))), "records": records}


def score_value(value: DefeatDiagnosisAnswer, item: DDItem) -> ScoreResult:
    """Score the diagnosis an answer submitted, whether it wrote it or a schema did."""
    diag: Dict = _blank_diagnostics(item)
    statuses = {str(s).upper() for s in value.get("status") or ()}
    pred: Dict[Tuple[str, str], set] = {}
    pred_surv: Dict[Tuple[str, str], str] = {}
    for rec in value.get("records") or ():
        # A record that names no defeater names no failure point, so it is not one
        # of the answer's claims and nothing is scored against it.
        if "defeated_at" not in rec or "defeater" not in rec:
            continue
        key = (rec["defeated_at"], rec["defeater"])
        pred.setdefault(key, set()).add(str(rec.get("kind", "")).lower())
        if "survives_because" in rec:
            pred_surv[key] = rec["survives_because"]
    diag["n_quoted"] = len(pred)
    if not statuses and not pred:
        # An empty submission lands here too. It is an answer with nothing in it, not
        # an answer that failed to arrive.
        return ScoreResult(0.0, False, "empty_answer", diag)

    diag["status_contradicted"] = len(statuses) > 1
    status_ok = statuses == {item.claim_status}
    diag["status_correct"] = status_ok
    diag["kind_contradicted"] = any(len(kinds) > 1 for kinds in pred.values())
    gold = {(d["defeated_at"], d["defeater"]): d["kind"] for d in item.diagnoses}
    matched = {k for k, kinds in pred.items() if kinds == {gold.get(k)}}
    tp = len(matched)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    diag["extra"] = sorted(str((a, b, k)) for (a, b), kinds in pred.items()
                           if (a, b) not in matched for k in kinds)[:4]
    diag["missing"] = sorted(str((a, b, k)) for (a, b), k in gold.items()
                             if (a, b) not in matched)[:4]

    gold_surv = {(d["defeated_at"], d["defeater"]): d["survives_because"]
                 for d in item.diagnoses if d.get("survives_because")}
    if gold_surv:
        hit = sum(1 for k, v in gold_surv.items() if k in matched and pred_surv.get(k) == v)
        surv_score = hit / len(gold_surv)
        diag["survives_because_correct"] = hit
        diag["survives_because_total"] = len(gold_surv)
    else:
        surv_score = None

    if surv_score is None:
        score_val = 0.85 * f1 + 0.15 * (1.0 if status_ok else 0.0)
    else:
        score_val = 0.60 * f1 + 0.15 * (1.0 if status_ok else 0.0) + 0.25 * surv_score
    exact_match = (tp == len(gold) == len(pred) and status_ok
                   and (surv_score is None or surv_score >= 0.999))
    diag.update(f1=round(f1, 4), precision=round(precision, 4), recall=round(recall, 4),
                status_correct=status_ok,
                survives_because_score=None if surv_score is None else round(surv_score, 4),
                exact_match=exact_match)
    return ScoreResult(round(score_val, 4), exact_match, "ok", diag)


def score(answer_text: str, item: DDItem) -> ScoreResult:
    """Score an answer written as text: read the value out of it, then score the value.

    A harness scoring a whole taskset wants a row for every item, so text that spells
    out no answer comes back as a zero rather than as an exception.
    """
    try:
        value = parse(answer_text, item)
    except UnparseableAnswer as e:
        diag: Dict = _blank_diagnostics(item)
        diag.update(e.diagnostics)
        return ScoreResult(0.0, False, e.reason, diag)
    return score_value(value, item)


def make_item_report(level: int, seed: int, ordering: str = LAST_LINK,
                     profile: str = "FULL", tries: int = 14) -> BuildReport:
    return retry(lambda k: build(level, seed * 83 + k, ordering, profile), tries)


def make_item(level: int, seed: int, ordering: str = LAST_LINK, profile: str = "FULL",
              tries: int = 14) -> Optional[DDItem]:
    return make_item_report(level, seed, ordering, profile, tries).item


PANEL_THRESHOLD = round(0.85 / 3 + 0.15 + 0.05, 3)
