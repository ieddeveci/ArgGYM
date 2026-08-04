from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier
from core.invariants import randomize_rule_names, split_atoms_and_rules

TASK = "defeat_diagnosis"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"
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


def build(level: int, seed: int, ordering: str = LAST_LINK) -> Optional[DDItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "dd"))
    n_routes = 1 if level <= EASY_LEVELS else 3
    depth = max(2, min(2 + level // 3, 7))
    tower = 0 if level < 5 else min(1 + (level - 5) // 4, 3)
    n_filler = max(0, min(2 + level * 2, 30))
    n_inert = 0 if level < 4 else min(1 + (level - 4) // 5, 3)

    names = _names(stable_seed(seed, level, ordering, "nm"),
                   40 + n_routes * (depth + 8) + n_filler * 2 + tower * 4)
    it = iter(names)
    claim = next(it)
    ops: List[Operation] = []
    ridx = [0]
    diagnoses: List[Dict] = []

    kinds = [UNDERMINE, UNDERCUT, REBUT]
    rng.shuffle(kinds)
    kind_seq = kinds[:n_routes]

    for k in range(n_routes):
        kind = kind_seq[k]
        use_axiom = (level >= 4 and kind != UNDERMINE and k % 2 == 0)
        strict_at = (depth // 2 + 1) if (level >= 6 and depth >= 3) else -1
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
        for j in range(depth):
            ridx[0] += 1
            nm = f"r_{ridx[0]}"
            nxt = claim if j == depth - 1 else next(it)
            is_strict = (j == strict_at and j != depth - 1)
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

    for _ in range(n_inert):
        ax = next(it)
        ops.append(Operation(kind="axiom", content=ax))
        mid = next(it)
        ridx[0] += 1
        rmid = f"r_{ridx[0]}"
        ops.append(Operation(kind="defeasible", name=rmid, antecedents=(ax,), consequent=mid))
        ridx[0] += 1
        rstrict = f"r_{ridx[0]}"
        ops.append(Operation(kind="strict", name=rstrict, antecedents=(mid,),
                             consequent=next(it)))
        q = next(it)
        ops.append(Operation(kind="premise", content=q))
        ridx[0] += 1
        ops.append(Operation(kind="defeasible", name=f"w_{ridx[0]}", antecedents=(q,),
                             consequent="-" + rstrict))

    for _ in range(n_filler):
        a, b = next(it), next(it)
        ops.append(Operation(kind="premise", content=a))
        ridx[0] += 1
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
        return None

    st = status(base, claim, ordering)
    if st not in ("OVERRULED", "UNDECIDED"):
        return None

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
        reference="[answer]\n" + "\n".join(lines) + "\n[/answer]",
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
    on = "the last-link strength ordering" if ordering == LAST_LINK \
        else "the weakest-link strength ordering"
    extra = ("   ...; survives_because: <rule>   -- append this where the defeater is itself "
             "attacked by a rule that is defeated\n" if want_survival else "")
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"The claim {claim} is not justified.\n"
            f"State its status, and identify every point at which its support fails.\n\n"
            "Answer format, between [answer] and [/answer]:\n"
            "   first line: `status: overruled` or `status: undecided`\n"
            "   then one line per failure point, as\n"
            "   `defeated_at: <target>; defeater: <defeater>; "
            "kind: undermine|undercut|rebut`\n"
            "   where <target> is the CLAIM attacked for undermine and rebut, "
            "and the RULE switched off for undercut, and <defeater> is the rule or "
            "asserted premise doing the attacking\n" + extra)


# Answer region comes from core.scoring: the LAST complete region, so a
# reasoning model that drafts and then revises is scored on the revision.
from core.scoring import answer_region
_STATUS = re.compile(r"status\s*[:=]\s*(justified|overruled|undecided)", re.I)
_FIELD = re.compile(r"(\w+)\s*[:=]\s*([^;\n]+)")


def score(answer_text: str, item: DDItem) -> Dict:
    diag: Dict = {"n_quoted": 0, "n_gold": len(item.diagnoses),
                  "status_correct": False, "extra": [], "missing": []}
    m = answer_region(answer_text)
    if m is None:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    body = m

    for _k in re.findall(r"kind\s*[:=]\s*([^;\n,]+)", body, flags=re.I):
        if _k.strip().strip(",;.").lower() not in (UNDERMINE, UNDERCUT, REBUT):
            diag["invalid_kind"] = _k.strip()
            return {"score": 0.0, "reason": f"invalid_kind:{_k.strip()[:16]}",
                    "diagnostics": diag}
    residue = _STATUS.sub(" ", body)
    residue = re.sub(r"(defeated_at|defeater|kind|survives_because)\s*[:=]\s*[^;\n]+",
                     " ", residue, flags=re.I)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    if junk:
        diag["n_unparseable"] = len(junk)
        diag["junk_tokens"] = junk[:6]
        return {"score": 0.0, "reason": f"unparseable_tokens:{len(junk)}",
                "diagnostics": diag}

    sm = _STATUS.search(body)
    status_ok = bool(sm) and sm.group(1).upper() == item.claim_status
    diag["status_correct"] = status_ok

    pred = set()
    pred_surv = {}
    last_key = None
    records = re.split(r"(?=defeated_at\s*[:=])", body)
    chunks = []
    for rec in records:
        if "defeated_at" in rec:
            chunks.append(rec)
        else:
            chunks.extend(rec.splitlines())
    for line in chunks:
        if _STATUS.search(line) and "defeated_at" not in line:
            continue
        fields = {k.lower(): v.strip().strip(",;.") for k, v in _FIELD.findall(line)}
        if "defeated_at" in fields and "defeater" in fields:
            key = (fields["defeated_at"], fields["defeater"], fields.get("kind", "").lower())
            pred.add(key)
            last_key = key
            if fields.get("survives_because"):
                pred_surv[key] = fields["survives_because"].strip()
        elif "survives_because" in fields and last_key is not None:
            pred_surv[last_key] = fields["survives_because"].strip()
    diag["n_quoted"] = len(pred)

    gold = {(d["defeated_at"], d["defeater"], d["kind"]) for d in item.diagnoses}
    tp = len(gold & pred)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    diag["extra"] = sorted(str(x) for x in pred - gold)[:4]
    diag["missing"] = sorted(str(x) for x in gold - pred)[:4]

    gold_surv = {(d["defeated_at"], d["defeater"], d["kind"]): d["survives_because"]
                 for d in item.diagnoses if d.get("survives_because")}
    if gold_surv:
        hit = sum(1 for k, v in gold_surv.items() if pred_surv.get(k) == v)
        surv_score = hit / len(gold_surv)
        diag["survives_because_correct"] = hit
        diag["survives_because_total"] = len(gold_surv)
    else:
        surv_score = None

    if surv_score is None:
        score_val = 0.85 * f1 + 0.15 * (1.0 if status_ok else 0.0)
    else:
        score_val = 0.60 * f1 + 0.15 * (1.0 if status_ok else 0.0) + 0.25 * surv_score
    return {"score": round(score_val, 4), "reason": "ok",
            "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "status_correct": status_ok,
            "survives_because_score": None if surv_score is None else round(surv_score, 4),
            "exact_match": (pred == gold and status_ok
                            and (surv_score is None or surv_score >= 0.999)),
            "diagnostics": diag}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              tries: int = 14) -> Optional[DDItem]:
    for k in range(tries):
        it = build(level, seed * 83 + k, ordering)
        if it is not None:
            return it
    return None


PANEL_THRESHOLD = round(0.85 / 3 + 0.15 + 0.05, 3)
