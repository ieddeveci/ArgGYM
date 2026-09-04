from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier
from arggym.core.curriculum import junction_budget, JUNCTION_CAPS, wants_ternary, junctions_for
from arggym.core.curriculum import negated_branch
from arggym.core.prompting import permitted_block
from arggym.core.invariants import (dedupe_parallel, minimal_subset_exact, assert_irredundant,
                        randomize_rule_names, remap_text, language_enrichment,
                        split_atoms_and_rules)

TASK = "counter_argument"
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


def _is_democratic(ordering: str) -> bool:
    return str(ordering).endswith("_democratic")
EASY_LEVELS = 4
_L = "abcdefghijklmnopqrstuvwxy"
# One pool per item, read through a single iterator by every consumer, so the chain,
# the decoy and the language enrichment can never be handed the same name. Two draws
# from two seeds could, and did. The pool covers the whole item: the heaviest cell on
# the export grid takes 60 names.
_POOL = 400


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def _names(seed: int, n: int) -> List[str]:
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:12] for b in _L[12:] for d in range(10)]
    rng.shuffle(pool)
    if n > len(pool):
        raise ValueError(f"name pool exhausted: asked {n} of {len(pool)}")
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


def status(ops: Sequence[Operation], lit: str, ordering: str) -> str:
    try:
        return str(ASPICVerifier.from_operations(list(ops), ordering=ordering).status(lit))
    except Exception:
        return "ERR"


@dataclass
class CAItem:
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    target: str
    ordering: str
    level: int
    reference: str
    min_directives: int
    seed_lit: str
    metadata: Dict = field(default_factory=dict)


def build(level: int, seed: int, ordering: str = LAST_LINK,
          allow_strict: bool = False) -> Optional[CAItem]:
    rng = random.Random(stable_seed(seed, level, ordering, "ca"))
    n_chain = max(1, min(1 + (level * 5) // 15, 6))
    depth = max(2, min(2 + (level * 3) // 15, 5))
    # Below level 6 this floors at one while there are only two or three chains, so a
    # strict-final chain always reached the target. A strict counter-argument then
    # contradicts it instead of winning, and the strict ablation had no cheap answer to
    # find at its own entry level: the cheapest item cost three directives where level 6
    # cost one (#32). Drawn here, so about half of those items admit the shortcut, which
    # is the same split levels 6 and up get from the mid-chain target.
    #
    # Only for the ablation. The plain variant asks a different question and the draw has
    # nothing to say about it, so gating keeps its items exactly as they were. The two
    # variants already build different theories below level 8, through `contested`.
    n_strict = 0 if level < 3 else min(n_chain, 1 + (level - 3) // 4)
    if allow_strict and 3 <= level < 6 and stable_seed(seed, level, ordering, "cas") % 2 == 0:
        n_strict = 0
    n_axiom_strict = 0
    use_decoy = level >= 9 and n_strict < n_chain
    contested = level >= 8 or allow_strict
    mid_target = level >= 6 and (seed % 2 == 1)

    it = iter(_names(stable_seed(seed, level, ordering, "nm"), _POOL))
    apex = next(it)
    seed_lit = next(it)
    ops: List[Operation] = [Operation(kind="premise", content=seed_lit)]

    if contested:
        for _ in range(2):
            c = next(it)
            ops.append(Operation(kind="premise", content=c))
            ops.append(Operation(kind="premise", content="-" + c))
            ops.append(Operation(kind="prefer_premise", stronger="-" + c, weaker=c))

    chains: List[Dict] = []
    ridx = 0
    strict_flags = [i < n_strict for i in range(n_chain)]
    rng.shuffle(strict_flags)
    junction = next(it) if (mid_target and depth >= 3 and n_chain >= 2) else None
    j_budget = junctions_for(level, max(1, n_chain * depth))
    for ci in range(n_chain):
        root = next(it)
        ax_strict = ci < n_axiom_strict
        ops.append(Operation(kind="axiom" if ax_strict else "premise", content=root))
        cur = root
        rules: List[Tuple[str, str, bool]] = []
        # Each junction branch rests on an ordinary premise of its own, and that premise
        # belongs to every argument the chain carries. Keyed by the branch rule so the
        # reference solution can slice premises and rules the same way; the democratic
        # paths below have to rank both. Remapped with the rule names after the rename.
        bprem: Dict[str, str] = {}
        tgt_idx: Optional[int] = None
        for j in range(depth):
            ridx += 1
            nm = f"d{ridx}"
            last = j == depth - 1
            if last:
                nxt = apex
            elif junction is not None and j == depth // 2 - 1:
                nxt = junction
            else:
                nxt = next(it)
            strict = ax_strict or (last and strict_flags[ci])
            _per = max(0, j_budget // max(1, n_chain))
            if j_budget and _per == 0 and ci < j_budget:
                _per = 1
            _jpts = set()
            if _per and depth >= 2:
                _stepj = max(1, (depth - 1) // (_per + 1))
                _jpts = {min(depth - 2, _stepj * (z + 1)) for z in range(_per)}
            if (not strict and not last and j in _jpts):
                _extra = []
                for _e in range(2 if wants_ternary(level, ci) else 1):
                    broot = next(it)
                    _bsrc = ("-" + broot) if negated_branch(ci) else broot
                    ops.append(Operation(kind="premise", content=_bsrc))
                    ridx += 1
                    bnm = f"d{ridx}"
                    blit = next(it)
                    ops.append(Operation(kind="defeasible", name=bnm, antecedents=(_bsrc,),
                                         consequent=blit))
                    rules.append((bnm, blit, False))
                    bprem[bnm] = _bsrc
                    _extra.append(blit)
                ridx += 1
                nm = f"d{ridx}"
                ops.append(Operation(kind="defeasible", name=nm,
                                     antecedents=tuple([cur] + _extra), consequent=nxt))
            else:
                ops.append(Operation(kind="strict" if strict else "defeasible", name=nm,
                                     antecedents=(cur,), consequent=nxt))
            rules.append((nm, nxt, strict))
            if junction is not None and j == depth // 2 - 1:
                tgt_idx = len(rules) - 1
            cur = nxt
        chains.append({"root": root, "rules": rules, "strict_final": strict_flags[ci],
                       "tgt_idx": tgt_idx, "bprem": bprem})

    if mid_target and depth >= 3 and n_chain >= 2:
        cand = junction
        supporters = [c for c in chains
                      if any(r[1] == cand for r in c["rules"])]
        if len(supporters) < n_chain:
            target = apex
            target_chains = chains
            target_rule_idx = None
        else:
            target = cand
            target_chains = supporters
            target_rule_idx = depth // 2 - 1
    else:
        target = apex
        target_chains = chains
        target_rule_idx = None

    def _tgt_idx(c: Dict) -> int:
        return c["tgt_idx"] if target_rule_idx is not None else -1

    def _upto_tgt(c: Dict) -> List[Tuple[str, str, bool]]:
        return c["rules"][:c["tgt_idx"] + 1] if target_rule_idx is not None else c["rules"]

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    ops = list(ops) + _lx
    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))

    if any(o.kind in ("premise", "axiom") and o.content == target for o in base):
        return None
    if status(base, target, ordering) != "JUSTIFIED":
        return None
    if status(base, "-" + target, ordering) == "JUSTIFIED":
        return None

    decoy_rule = None
    decoy_root = None
    if use_decoy:
        droot = next(it)
        decoy_root = droot
        ops.append(Operation(kind="premise", content=droot))
        ridx += 1
        decoy_rule = f"d{ridx}"
        ops.append(Operation(kind="defeasible", name=decoy_rule, antecedents=(droot,),
                             consequent="-" + target))
        cand = [c for c in target_chains if not c["rules"][_tgt_idx(c)][2]]
        killer = cand[0]["rules"][_tgt_idx(cand[0])] if cand else None
        if killer is None:
            decoy_rule = None
        else:
            ops.append(Operation(kind="prefer_rule", stronger=killer[0], weaker=decoy_rule))
        base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))
        if decoy_rule and status(base, "-" + target, ordering) == "JUSTIFIED":
            return None

    ops, _rmap = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    for c in chains:
        c["rules"] = [(_rmap.get(nm, nm), lit, st) for nm, lit, st in c["rules"]]
        c["bprem"] = {_rmap.get(nm, nm): lit for nm, lit in c["bprem"].items()}
    if decoy_rule is not None:
        decoy_rule = _rmap.get(decoy_rule, decoy_rule)
    base = _ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))
    atoms, rnames = split_atoms_and_rules(base)
    if atoms & rnames:
        return None

    pairs: List[Tuple[Operation, str]] = []

    def add(op: Operation, line: str) -> None:
        if line not in {l for _o, l in pairs}:
            pairs.append((op, line))

    if decoy_rule:
        killer = next((c["rules"][_tgt_idx(c)] for c in target_chains
                       if not c["rules"][_tgt_idx(c)][2]), None)
        if killer is None:
            return None
        add(Operation(kind="defeasible", name="z0", antecedents=(seed_lit,),
                      consequent="-" + killer[0]),
            f"[defeasible z0: {seed_lit} => -{killer[0]}]")
        for ci, c in enumerate(target_chains):
            rl = c["rules"][_tgt_idx(c)]
            if rl[0] == killer[0]:
                continue
            if rl[2]:
                first = c["rules"][0]
                add(Operation(kind="defeasible", name=f"z{ci+1}", antecedents=(seed_lit,),
                              consequent="-" + first[0]),
                    f"[defeasible z{ci+1}: {seed_lit} => -{first[0]}]")
            else:
                add(Operation(kind="prefer_rule", stronger=decoy_rule, weaker=rl[0]),
                    f"[prefer_rule: {decoy_rule} > {rl[0]}]")
                if _is_weakest(ordering):
                    for rr in _upto_tgt(c):
                        if not rr[2]:
                            add(Operation(kind="prefer_rule", stronger=decoy_rule,
                                          weaker=rr[0]),
                                f"[prefer_rule: {decoy_rule} > {rr[0]}]")
                        # Democratic reads the whole set: the chain is weaker only when
                        # every element of it is weaker than something in the decoy, so a
                        # branch premise left unranked keeps the chain incomparable and it
                        # defeats the decoy back. Elitist needs one weak element and gets
                        # it from the root below, so ranking the branch there only enlarges
                        # the subset search.
                        _bp = c["bprem"].get(rr[0]) if _is_democratic(ordering) else None
                        if _bp is not None:
                            add(Operation(kind="prefer_premise", stronger=decoy_root,
                                          weaker=_bp),
                                f"[prefer_premise: {decoy_root} > {_bp}]")
                    add(Operation(kind="prefer_premise", stronger=decoy_root,
                                  weaker=c["root"]),
                        f"[prefer_premise: {decoy_root} > {c['root']}]")
    else:
        add(Operation(kind="defeasible", name="w", antecedents=(seed_lit,),
                      consequent="-" + target),
            f"[defeasible w: {seed_lit} => -{target}]")
        for ci, c in enumerate(target_chains):
            rules_upto = _upto_tgt(c)
            final = rules_upto[-1]
            if final[2]:
                first = c["rules"][0]
                add(Operation(kind="defeasible", name=f"z{ci}", antecedents=(seed_lit,),
                              consequent="-" + first[0]),
                    f"[defeasible z{ci}: {seed_lit} => -{first[0]}]")
                continue
            if _is_last(ordering):
                add(Operation(kind="prefer_rule", stronger="w", weaker=final[0]),
                    f"[prefer_rule: w > {final[0]}]")
            else:
                add(Operation(kind="prefer_premise", stronger=seed_lit, weaker=c["root"]),
                    f"[prefer_premise: {seed_lit} > {c['root']}]")
                for rl in rules_upto:
                    if not rl[2]:
                        add(Operation(kind="prefer_rule", stronger="w", weaker=rl[0]),
                            f"[prefer_rule: w > {rl[0]}]")
                    # Same as the decoy path above: democratic needs every element of the
                    # chain ranked, branch premises included.
                    _bp = c["bprem"].get(rl[0]) if _is_democratic(ordering) else None
                    if _bp is not None:
                        add(Operation(kind="prefer_premise", stronger=seed_lit, weaker=_bp),
                            f"[prefer_premise: {seed_lit} > {_bp}]")

    ref_ops, lines = dedupe_parallel(pairs)

    def holds(ops_subset) -> bool:
        p2 = [o for o in ops_subset if o.kind not in ("prefer_rule", "prefer_premise")]
        f2 = [o for o in ops_subset if o.kind in ("prefer_rule", "prefer_premise")]
        try:
            v = ASPICVerifier.from_operations(base + p2 + f2, ordering=ordering)
            return (str(v.status("-" + target)) == "JUSTIFIED"
                    and str(v.status(target)) == "OVERRULED"
                    and v.is_consistent())
        except Exception:
            return False

    best, _proven, _calls = minimal_subset_exact(ref_ops, holds, max_calls=40000)
    if best is None:
        return None
    keep = {id(o) for o in best}
    pairs = [(o, l) for o, l in zip(ref_ops, lines) if id(o) in keep]
    ref_ops, lines = dedupe_parallel(pairs)
    if not ref_ops or not holds(ref_ops):
        return None
    irredundant, _irr_calls = assert_irredundant(ref_ops, holds)

    bank: List[Tuple[List[Operation], List[str]]] = []
    if allow_strict:
        bank.append(([Operation(kind="strict", name="cs", antecedents=(seed_lit,),
                                consequent="-" + target)],
                     [f"[strict cs: {seed_lit} -> -{target}]"]))
    for c in target_chains:
        rl = c["rules"][_tgt_idx(c)]
        if not rl[2]:
            bank.append(([Operation(kind="defeasible", name="cw", antecedents=(seed_lit,),
                                    consequent="-" + target),
                          Operation(kind="prefer_rule", stronger="cw", weaker=rl[0])],
                         [f"[defeasible cw: {seed_lit} => -{target}]",
                          f"[prefer_rule: cw > {rl[0]}]"]))
    for cand_ops, cand_lines in bank:
        if len(cand_lines) >= len(lines):
            continue
        if holds(cand_ops):
            ref_ops, lines = cand_ops, cand_lines
            break

    prompt = _render_prompt(render_ops(base), target, ordering, allow_strict)
    return CAItem(
        prompt=prompt, theory_text=render_ops(base), base_ops=base, target=target,
        ordering=ordering, level=level,
        reference="\n".join(lines),
        min_directives=len(lines), seed_lit=seed_lit,
        metadata={
            "allow_strict": allow_strict, "n_axiom_strict_chains": n_axiom_strict,
            "n_chains": n_chain, "chain_depth": depth,
            "n_strict_final": sum(1 for c in chains if c["strict_final"]),
            "mid_chain_target": target_rule_idx is not None,
            "decoy_present": bool(decoy_rule), "contested_seed": contested,
            "n_rules": len([o for o in base if o.kind in ("defeasible", "strict")]),
            "n_atoms": len(atoms),
            "reference_directives": len(lines),
            "strategy": "revive_decoy" if decoy_rule else "build",
            "reference_irredundant": irredundant,
            "irredundance_calls": _irr_calls,
            "irredundance_budget_exhausted": irredundant is None,
            "minimality_proven": _proven,
            "min_within_reference": len(lines),
        })


def _render_prompt(theory: str, target: str, ordering: str,
                   allow_strict: bool = False) -> str:
    on = _ordering_phrase(ordering)
    return (f"The following is a defeasible argumentation theory, evaluated under grounded semantics "
            f"with {on}.\n\n{theory}\n\n"
            f"The claim {target} is currently justified.\n"
            f"What is the minimal set of directives that makes -{target} justified "
            f"and {target} overruled?\n\n"
            f"{permitted_block(allow_strict)}\n\n"
            "Answer format: one directive per line.")


def as_score_input(it: "CAItem") -> Dict:
    return {"base_ops": it.base_ops, "ordering": it.ordering,
            "goals": [{"claim": "-" + it.target, "want": "JUSTIFIED"},
                      {"claim": it.target, "want": "OVERRULED"}],
            "min_directives": it.min_directives,
            "allow_strict": it.metadata.get("allow_strict", False)}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              tries: int = 14, allow_strict: bool = False) -> Optional[CAItem]:
    for k in range(tries):
        it = build(level, seed * 53 + k, ordering, allow_strict)
        if it is not None:
            return it
    return None
