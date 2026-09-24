from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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
from arggym.core.invariants import (
    language_enrichment,
    ordered_ops,
    randomize_rule_names,
    split_atoms_and_rules,
)
from arggym.core.prompting import STRAY_TEXT, answer_format
from arggym.core.scoring import unmark

TASK = "claim_chain"
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
    if n > len(pool):
        raise ValueError(f"name pool exhausted: asked {n}, have {len(pool)}")
    return pool[:n]


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
class CCItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    claim: str
    line_ops: List[Operation]
    ordering: str
    level: int
    reference: str
    metadata: Dict = field(default_factory=dict)


def _tower(ops: List[Operation], names, ridx: List[int], attacked_lit: str,
           height: int) -> None:
    prev_rule = None
    for i in range(height):
        root = next(names)
        ops.append(Operation(kind="premise", content=root))
        ridx[0] += 1
        nm = f"w_{ridx[0]}"
        cons = ("-" + attacked_lit) if i == 0 else ("-" + prev_rule)
        ops.append(Operation(kind="defeasible", name=nm, antecedents=(root,), consequent=cons))
        prev_rule = nm


def _root(ops: List[Operation], names, negated: bool) -> str:
    """Add the premise a chain starts from. A negated root is attacked by its positive
    twin and survives on the premise preference."""
    base = next(names)
    ops.append(Operation(kind="premise", content=base))
    if not negated:
        return base
    ops.append(Operation(kind="premise", content="-" + base))
    ops.append(Operation(kind="prefer_premise", stronger="-" + base, weaker=base))
    return "-" + base


def _chain(ops: List[Operation], names, ridx: List[int], root: str, claim: str,
           depth: int, j_points: Set[int], level: int
           ) -> Tuple[List[Operation], List[str], List[str]]:
    """A chain of `depth` rules from `root` to `claim`, with a junction at each of
    `j_points`. Returns its directives, its trunk rule names and its trunk literals."""
    line_ops = [Operation(kind="premise", content=root)]
    trunk_rules: List[str] = []
    trunk_lits: List[str] = []
    order = sorted(j_points)
    cur = root
    for j in range(depth):
        ridx[0] += 1
        nm = f"r_{ridx[0]}"
        nxt = claim if j == depth - 1 else next(names)
        extra_lits = []
        if j in j_points:
            for _e in range(2 if wants_ternary(level, order.index(j)) else 1):
                broot = next(names)
                src = ("-" + broot) if negated_branch(order.index(j) * 2 + _e) else broot
                ops.append(Operation(kind="premise", content=src))
                ridx[0] += 1
                brule = Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(src,),
                                  consequent=next(names))
                ops.append(brule)
                extra_lits.append(brule.consequent)
                line_ops.append(Operation(kind="premise", content=src))
                line_ops.append(brule)
            ridx[0] += 1
            nm = f"r_{ridx[0]}"
        r = Operation(kind="defeasible", name=nm, antecedents=tuple([cur] + extra_lits),
                      consequent=nxt)
        ops.append(r)
        line_ops.append(r)
        trunk_rules.append(nm)
        trunk_lits.append(nxt)
        cur = nxt
    return line_ops, trunk_rules, trunk_lits


def build(level: int, seed: int, ordering: str = LAST_LINK,
          profile: str = "FULL") -> Union[CCItem, Rejected]:
    depth = max(2, min(2 + level, 20))
    n_decoy = 1 if level < 4 else min(1 + (level - 4) // 4, 3)
    n_filler = max(0, min(4 + level * 3, 60))
    branch_decoys = level >= 11
    j_budget = min(3, junction_budget(level, JUNCTION_CAPS["claim_chain"]))
    use_neg_root = level >= 4

    # From level 4 every derivation of the claim is attacked by one tower: a rule
    # rebutting one of its trunk literals, and rules above it each undercutting the one
    # below. The line's tower has even height, so its bottom rule ends up defeated and
    # the line stands. A decoy's has odd height of at least 3, so its bottom rule is
    # undercut too, yet stands, and the decoy falls: only walking the whole tower
    # decides (#185). Every tower's target is drawn the same way. Heights grow in
    # stages: 2 against 3 at levels 4-7, where the shorter tower is the even one and
    # reading its length is reading its parity; from level 8 the line's is 2 or 4
    # (4 or 6 from level 12) with each decoy's one shorter or one longer, and how many
    # decoy towers are shorter than the line's is uniform over 0..n_decoy, so the
    # line is the shortest or the longest no more often than any other derivation.
    tw = random.Random(stable_seed(seed, level, ordering, "tw"))
    tower_true, decoy_towers = 0, [0] * n_decoy
    if use_neg_root and level < 8:
        tower_true, decoy_towers = 2, [3] * n_decoy
    elif use_neg_root:
        extra = 0 if level < 12 else 2
        n_short = tw.randrange(n_decoy + 1)
        tower_true = (2 if n_short == 0 else 4) + extra
        short = set(tw.sample(range(n_decoy), n_short))
        decoy_towers = [tower_true - 1 if k in short else tower_true + 1
                        for k in range(n_decoy)]

    names = _names(stable_seed(seed, level, ordering, "nm"),
                   ((60 + depth * 3 + n_decoy * (depth + 6) * 4) + 6 * j_budget) + n_filler * 2
                   + tower_true + sum(decoy_towers))
    it = iter(names)
    claim = next(it)
    ops: List[Operation] = []
    ridx = [0]

    j_points = set()
    if j_budget and depth >= 3:
        step = max(1, depth // (j_budget + 1))
        j_points = {min(depth - 2, step * (i + 1)) for i in range(j_budget)}

    root = _root(ops, it, use_neg_root)
    line_ops, _, true_lits = _chain(ops, it, ridx, root, claim, depth, j_points, level)
    chains = [line_ops]
    if tower_true:
        _tower(ops, it, ridx, true_lits[tw.randrange(depth - 1)], tower_true)

    # A decoy is built exactly like the true line -- same root sign and root preference,
    # same junctions, same branch signs -- so nothing but the attacks tells them apart
    # (#185). Below level 4 the line is unattacked and each decoy fails to one attack.
    decoy_info = []
    for k in range(n_decoy):
        droot = _root(ops, it, use_neg_root)
        dline, drules, dlits = _chain(ops, it, ridx, droot, claim, depth, j_points, level)
        chains.append(dline)
        mode = (k + level) % 4
        if decoy_towers[k]:
            step = tw.randrange(depth - 1)
            _tower(ops, it, ridx, dlits[step], decoy_towers[k])
            where = f"tower of {decoy_towers[k]} at step {step + 1}"
        elif mode == 3:
            ops.append(Operation(
                kind="axiom" if PROFILES[profile].permits("axiom") else "premise",
                content="-" + droot))
            where = "impossible-root"
        elif mode == 1:
            _tower(ops, it, ridx, dlits[len(dlits) // 2], 1)
            where = "mid"
        else:
            src = next(it)
            ops.append(Operation(kind="premise", content=src))
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"w_{ridx[0]}", antecedents=(src,),
                                 consequent="-" + drules[-1]))
            where = "near-claim"
        decoy_info.append({"rules": drules, "defeat_at": where})

    # Two rules leading nowhere hang off each derivation at the same place, so "the one
    # whose literals other rules build on" is every derivation rather than the line.
    if branch_decoys and depth >= 4:
        for chain in chains:
            _rules = [o for o in chain if o.kind in ("defeasible", "strict") and o.consequent]
            if not _rules:
                return Rejected("no_line_rule_to_anchor")
            cur = _rules[min(len(_rules) - 1, depth // 3)].consequent
            for j in range(2):
                ridx[0] += 1
                nxt = next(it)
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                     antecedents=(cur,), consequent=nxt))
                cur = nxt

    _base = sum(1 for o in ops if o.kind == "defeasible")
    _have = sum(1 for o in ops if o.kind == "defeasible" and len(o.antecedents or ()) > 1)
    _fill_j = 0
    for _try in range(0, n_filler + 1):
        _total = _base + n_filler + _try * 2
        if (_have + _try) >= junctions_for(level, _total, solve=False):
            _fill_j = _try
            break
    else:
        _fill_j = n_filler
    for _fi in range(n_filler):
        a, b = next(it), next(it)
        ops.append(Operation(kind="premise", content=a))
        ridx[0] += 1
        if _fi < _fill_j:
            _ex = []
            for _e in range(2 if (level >= 9 and _fi % 2 == 0) else 1):
                br, bl = next(it), next(it)
                _src = ("-" + br) if negated_branch(_fi * 2 + _e) else br
                ops.append(Operation(kind="premise", content=_src))
                ridx[0] += 1
                ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                     antecedents=(_src,), consequent=bl))
                _ex.append(bl)
            ridx[0] += 1
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}",
                                 antecedents=tuple([a] + _ex), consequent=b))
        else:
            ops.append(Operation(kind="defeasible", name=f"r_{ridx[0]}", antecedents=(a,),
                                 consequent=b))

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    ops = list(ops) + _lx
    ops, _map = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    _line_roots = {x.content for x in line_ops if x.kind == "premise"}
    _line_rules = {_map.get(x.name, x.name) for x in line_ops if getattr(x, "name", None)}
    line_ops = [o for o in ops
                if (o.kind in ("premise", "axiom") and o.content in _line_roots)
                or (o.kind in ("defeasible", "strict") and o.name in _line_rules)]
    base = ordered_ops(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))

    atoms, rnames = split_atoms_and_rules(base)
    if atoms & rnames:
        return Rejected("atom_rule_name_collision")
    if status(base, claim, ordering) != "JUSTIFIED":
        return Rejected("claim_not_justified")

    for d in decoy_info:
        sub = [o for o in base if not (o.kind == "defeasible" and o.name in d["rules"])]
        if status(sub, claim, ordering) != "JUSTIFIED":
            return Rejected("decoy_is_load_bearing")
    without_true = [o for o in base
                    if not (o.kind == "defeasible" and o.name == line_ops[1].name)]
    if status(without_true, claim, ordering) == "JUSTIFIED":
        return Rejected("line_rule_not_necessary")

    ref_lines = [render_op(o) for o in line_ops]
    prompt = _render_prompt(render_ops(base), claim, ordering)
    return CCItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, claim=claim,
        line_ops=line_ops, ordering=ordering, level=level,
        reference="\n".join(ref_lines),
        metadata={
            "chain_depth": depth, "n_decoys": n_decoy, "tower_height_true": tower_true,
            "decoy_defeat_points": [d["defeat_at"] for d in decoy_info],
            "branch_decoys": branch_decoys,
            "n_items": len(base), "line_length": len(line_ops),
            "negated_line_root": use_neg_root,
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
        })


def _render_prompt(theory: str, claim: str, ordering: str) -> str:
    on = _ordering_phrase(ordering)
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"The claim {claim} is justified.\n"
            f"Write all and only the directives that form the argumentation line justifying {claim}, "
            "in order from the premise to the claim.\n\n"
            + answer_format("Answer format: one directive per line, copied exactly as it "
                            "appears above.\n" + STRAY_TEXT))


def _in_support_order(picked: Sequence[Operation]) -> Tuple[int, int]:
    """Of the rules quoted, how many arrive after everything they rest on.

    "In order from the premise to the claim" is a constraint, not one sequence. From
    level 6 the line is a tree rather than a chain -- six premises feeding sixteen rules
    at level 9 -- and its branches interleave in billions of ways that all read premise
    to claim. Matching against the order the generator happened to append in would score
    the branch order rather than the reasoning, so the check is the property itself:
    every antecedent of a rule is introduced by an earlier line of the answer.
    """
    have: Set[str] = set()
    n_rules = ok = 0
    for o in picked:
        if o.kind in ("premise", "axiom"):
            have.add(o.content)
            continue
        n_rules += 1
        if all(a in have for a in o.antecedents):
            ok += 1
        if o.consequent:
            have.add(o.consequent)
    return ok, n_rules


#: What a `claim_chain` answer is: the directives of the justifying line, as the
#: answer wrote them. The prompt asks for them "in order from the premise to the
#: claim" and the score reads that order, so the value is a sequence and not a set.
ClaimChainAnswer = List[str]

_QUOTED = re.compile(r"\[[^\]]*\]")


def _blank_diagnostics(item: CCItem) -> Dict:
    """The keys every result carries, so a zero and a one have the same shape."""
    return {"n_quoted": 0, "n_gold": len(item.line_ops), "extra": [], "missing": []}


def parse(text: str, item: CCItem) -> ClaimChainAnswer:
    """The directive lines an answer quotes, in the order it wrote them.

    The text arrives already extracted: composing the prompt and pulling the answer
    out of whatever came back is the harness's job, so the dataset never strips the
    answer delimiters. That is what lets a caller use any convention at all -- or none, with a
    solver that submits the value directly (`docs/dataset-contract.md` section 4).

    An answer with no brackets is read a line at a time, so a solver that drops the
    delimiters still submits something scorable. An answer that has brackets and
    also prose between them is refused: the brackets say it meant to quote, so the
    prose is not a second convention but text the answer did not account for.

    An empty answer parses to an empty sequence. It is an answer with nothing in
    it, not an answer that failed to arrive.
    """
    body = unmark(text)
    quoted = _QUOTED.findall(body)
    if not quoted:
        return [line.strip() for line in body.splitlines() if line.strip()]
    residue = _QUOTED.sub(" ", body)
    junk = [t for t in residue.split()
            if t.strip(",;.-*\u2022()") and not re.fullmatch(r"\d+[.)]?", t)]
    if junk:
        raise UnparseableAnswer(f"unparseable_tokens:{len(junk)}",
                                {"n_unparseable": len(junk), "junk_tokens": junk[:6]})
    return quoted


def score_value(value: ClaimChainAnswer, item: CCItem) -> ScoreResult:
    """Score the directives an answer submitted, whether it wrote them or a schema did."""
    quoted = list(value)
    diag: Dict = _blank_diagnostics(item)
    diag["n_quoted"] = len(quoted)
    if not quoted:
        return ScoreResult(0.0, False, "empty_answer", diag)

    gold_lines = [render_op(o) for o in item.line_ops]
    gold_set, pred_set = set(gold_lines), set(quoted)
    # Content is still a set: which directives the line is made of. Counting the answer's
    # lines rather than its distinct lines is new, so a repeated directive is not free.
    tp = len(gold_set & pred_set)
    precision = tp / max(len(quoted), 1)
    recall = tp / max(len(gold_set), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    diag["extra"] = sorted(pred_set - gold_set)[:5]
    diag["missing"] = sorted(gold_set - pred_set)[:5]

    by_line = {render_op(o): o for o in item.base_ops}
    picked = [by_line[l] for l in quoted if l in by_line]

    # The order term. The prompt requires the line premise-to-claim and the score ignored
    # it, so reversing the complete gold answer scored 1.000 (#25).
    _ok, _n_rules = _in_support_order(picked)
    order_factor = 1.0 if not _n_rules else _ok / _n_rules
    diag["n_rules_quoted"] = _n_rules
    diag["n_rules_after_their_antecedents"] = _ok
    f1 *= order_factor
    justifies = False
    if picked:
        alone = status(picked, item.claim, item.ordering) == "JUSTIFIED"
        if alone:
            try:
                v = ASPICVerifier.from_operations(list(item.base_ops), ordering=item.ordering)
                lits = [o.consequent for o in picked
                        if o.kind in ("defeasible", "strict") and o.consequent]
                justifies = (str(v.status(item.claim)) == "JUSTIFIED"
                             and all(str(v.status(x)) == "JUSTIFIED" for x in lits))
            except Exception:
                justifies = False
    diag["behavioural_in_theory"] = justifies

    # "All and only the directives" is set equality; "in order from the premise to the
    # claim" is the order constraint. `behaviourally_justifies` is deliberately out of the
    # conjunction: it is computed on the resolved lines only, so it holds for an answer
    # that also carries junk (docs/dataset-contract.md, section 5).
    exact_match = pred_set == gold_set
    correct_order = order_factor == 1.0
    diag.update(f1=round(f1, 4), precision=round(precision, 4), recall=round(recall, 4),
                exact_match=exact_match, correct_order=correct_order,
                matches_reference_order=quoted == gold_lines,
                order_factor=round(order_factor, 4),
                behaviourally_justifies=justifies)
    return ScoreResult(round(f1, 4), exact_match and correct_order, "ok", diag)


def score(answer_text: str, item: CCItem) -> ScoreResult:
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
    return retry(lambda k: build(level, seed * 71 + k, ordering, profile), tries)


def make_item(level: int, seed: int, ordering: str = LAST_LINK, profile: str = "FULL",
              tries: int = 14) -> Optional[CCItem]:
    return make_item_report(level, seed, ordering, profile, tries).item
