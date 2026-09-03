from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier

from arggym.structures.chains import CONFIGS, LAST_LINK, WEAKEST_LINK, reasoning_cost
from arggym.core.invariants import randomize_rule_names, remap_text, language_enrichment
from arggym.core.minimality import find_minimum, find_minimum_decomposed
from arggym.structures.defence import (build_defence, verify_minimum as verify_defence_minimum,
                     verify_minimum_decomposed as verify_defence_decomposed)
from arggym.structures.interaction import build_mixed, check_interference, solve_mixed
from arggym.core.prompting import render
from arggym.core.curriculum import PROFILES, ATTACK, DEFENCE, MIXED, spec_for
from arggym.core.curriculum import junction_budget, JUNCTION_CAPS, wants_ternary, junctions_for
from arggym.core.scoring import score_item

TASK = "attack_defense"
_L = "abcdefghijklmnopqrstuvwxy"


def stable_seed(*parts) -> int:
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


_POOL = 400


def _names(seed: int, n: int) -> List[str]:
    import random
    rng = random.Random(seed)
    pool = [f"{a}{b}{d}" for a in _L[:12] for b in _L[12:] for d in range(10)]
    rng.shuffle(pool)
    return pool[:n]


@dataclass
class Item:
    task: str
    level: int
    ordering: str
    mode: str
    prompt: str
    theory_text: str
    base_ops: List[Operation]
    goals: List[Dict]
    reference: str
    min_directives: int
    metadata: Dict = field(default_factory=dict)

    def as_score_input(self) -> Dict:
        return {"base_ops": self.base_ops, "ordering": self.ordering,
                "goals": [{"claim": g["claim"], "want": g["want"]} for g in self.goals],
                "min_directives": self.min_directives,
                "subgoals": self.subgoals}

    @property
    def subgoals(self) -> List[str]:
        out = []
        for g in self.goals:
            if g.get("want") != "OVERRULED":
                continue
            for o in self.base_ops:
                if o.kind in ("defeasible", "strict") and o.consequent == g["claim"]:
                    for a in (o.antecedents or ()):
                        if a not in out:
                            out.append(a)
        return out


def _ops_ordered(ops: Sequence[Operation], shuffle_seed: Optional[int] = None) -> List[Operation]:
    facts = [o for o in ops if o.kind in ("premise", "axiom")]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    prefs = [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
    if shuffle_seed is not None:
        import random
        rng = random.Random(shuffle_seed)
        rng.shuffle(facts)
        rng.shuffle(rules)
    return facts + rules + prefs


def _render_ops(ops: Sequence[Operation]) -> str:
    out = []
    for o in _ops_ordered(ops):
        if o.kind in ("premise", "axiom"):
            out.append(f"[{o.kind}: {o.content}]")
        elif o.kind in ("defeasible", "strict"):
            arrow = "=>" if o.kind == "defeasible" else "->"
            out.append(f"[{o.kind} {o.name}: {' AND '.join(o.antecedents)} {arrow} {o.consequent}]")
        else:
            out.append(f"[{o.kind}: {o.stronger} > {o.weaker}]")
    return "\n".join(out)


def _atoms_and_rules(ops: Sequence[Operation]) -> Tuple[set, set]:
    atoms, rules = set(), set()
    for o in ops:
        if o.kind in ("premise", "axiom"):
            atoms.add(o.content.lstrip("-"))
        elif o.kind in ("defeasible", "strict"):
            if o.name:
                rules.add(o.name)
            for a in o.antecedents or ():
                atoms.add(a.lstrip("-"))
    return atoms, rules


def build_attack_item(level: int, seed: int, ordering: str,
                      profile: str = "FULL") -> Optional[Item]:
    import random
    sp = spec_for(level, ordering, variant=seed % 5)
    rng = random.Random(stable_seed(seed, level, ordering, "atkmix"))
    n = max(2, min(sp.n_chains_max, 2 + level // 3))
    depth = max(2, min(sp.depth_max, 2 + level // 4))

    pool = ["C2"] if level <= 3 else (["C2", "C4"] if level <= 6 else
                                      ["C2", "C4", "C7"] if level <= 9 else
                                      ["C2", "C4", "C6", "C7", "C8", "C9"])
    prof = PROFILES[profile]
    if not prof.permits("axiom"):
        pool = [c for c in pool if c not in ("C3", "C4")] or ["C2"]
    picks = [pool[i % len(pool)] for i in range(n)]
    rng.shuffle(picks)
    want_distinct = 1 if level <= 3 else (2 if level <= 9 else 3)
    if len(set(picks)) < min(want_distinct, len(pool)):
        return None

    it = iter(_names(stable_seed(seed, level, ordering, "atk"), _POOL))
    target = next(it)
    src = next(it)
    ops: List[Operation] = [Operation(kind="premise", content=src)]

    n_decoy_strict = 0 if level < 4 else min(1 + (level - 4) // 4, 3)

    survival = level >= 9
    decoy_srcs: List[str] = []
    if survival:
        for _ in range(2):
            d = next(it)
            ops.append(Operation(kind="premise", content=d))
            ops.append(Operation(kind="premise", content="-" + d))
            ops.append(Operation(kind="prefer_premise", stronger="-" + d, weaker=d))
            decoy_srcs.append(d)
    chains = []
    ridx = 0
    j_budget = junctions_for(level, max(1, n * depth))
    for ci, cname in enumerate(picks):
        cs = CONFIGS[cname]
        root = next(it)
        mids = [next(it) for _ in range(depth + 2)]
        rn = []
        for _ in range(depth + 2):
            ridx += 1
            rn.append(f"d{ridx}")
        per_chain = max(0, j_budget // max(1, len(picks)))
        if j_budget and per_chain == 0 and ci < j_budget:
            per_chain = 1
        if cname == "C7":
            k = max(1, min(depth - 1, 2))
            ch = cs.builder(root, mids, target, rn, depth, k,
                            n_junctions=per_chain, ternary=wants_ternary(level, ci))
        else:
            ch = cs.builder(root, mids, target, rn, depth,
                            n_junctions=per_chain, ternary=wants_ternary(level, ci))
        pass
        chains.append(ch)
        ops.extend(ch.to_ops())
    if not any(c.rules[-1].get("strict") for c in chains):
        return None
    for j in range(n_decoy_strict):
        dr, dm, dc = next(it), next(it), next(it)
        ridx += 1
        ops.append(Operation(kind="premise", content=dr))
        ops.append(Operation(kind="defeasible", name=f"d{ridx}", antecedents=(dr,), consequent=dm))
        ridx += 1
        ops.append(Operation(kind="strict", name=f"d{ridx}", antecedents=(dm,), consequent=dc))

    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    ops = list(ops) + _lx
    ops, _rmap = randomize_rule_names(ops, stable_seed(seed, level, ordering, "rn"))
    for _c in chains:
        for _r in _c.rules:
            _r["name"] = _rmap.get(_r["name"], _r["name"])
    base = _ops_ordered(ops, shuffle_seed=stable_seed(seed, level, ordering, "shuf"))

    atoms, rnames = _atoms_and_rules(base)
    if atoms & rnames:
        return None
    try:
        v = ASPICVerifier.from_operations(base, ordering=ordering)
        if str(v.status(target)) != "JUSTIFIED":
            return None
    except Exception:
        return None

    n_ch = len(chains)
    if n_ch >= 5:
        mn = find_minimum_decomposed(base, chains, target, src, "OVERRULED", ordering)
        if not mn.get("found"):
            mn = find_minimum(base, chains, target, src, "OVERRULED", ordering,
                              exhaustive_limit=3000)
    else:
        mn = find_minimum(base, chains, target, src, "OVERRULED", ordering,
                          exhaustive_limit=3000)
    if not mn["found"]:
        mn = find_minimum_decomposed(base, chains, target, src, "OVERRULED", ordering)
    if not mn.get("found") or mn.get("required_moves") is None:
        return None

    ref_lines = []
    for i, c in enumerate(chains):
        dfs = c.defeasible_rules()
        if not dfs:
            return None
        ref_lines.append(f"[defeasible k{i+1}: {src} => -{dfs[0]['name']}]")
    ref = remap_text("[answer]\n" + "\n".join(ref_lines) + "\n[/answer]", _rmap)

    goals = [{"claim": target, "current": "JUSTIFIED", "want": "OVERRULED"}]
    item = Item(task=TASK, level=level, ordering=ordering, mode=ATTACK,
                prompt=render(_render_ops(base), ordering, goals),
                theory_text=_render_ops(base), base_ops=base, goals=goals,
                reference=ref, min_directives=mn["required_moves"],
                metadata={
                    "n_chains": n, "chain_depth": depth, "configs": picks,
                    "distinct_configs": len(set(picks)),
                    "n_chain_rules": sum(len(c.rules) for c in chains),
                    "n_theory_rules": sum(1 for o in base
                                          if o.kind in ("defeasible", "strict")),
                    "n_rules": sum(len(c.rules) for c in chains),
                    "min_moves": mn["required_moves"],
                    "rejected_moves": mn["rejected_moves"],
                    "moves_available": mn["n_moves_available"],
                    "lower_bound_proven": mn["lower_bound_is_n_chains"],
                    "searched_exhaustively": mn["searched_exhaustively"],
                    "forced_moves": mn.get("forced_moves"),
                    "combos_checked": mn.get("combos_checked"),
                    "minimality_proven": mn.get("minimality_proven"),
                    "search_floor": mn.get("search_floor"),
                    "per_chain_cost": [reasoning_cost(c, ordering)["cost"] for c in chains],
                    "per_chain_admits": [sorted(CONFIGS[c].admits(ordering)) for c in picks],
                    "n_decoy_strict": n_decoy_strict,
                    "shuffled_presentation": True, "profile": profile,
                    "survival": survival,
                    "contested_premises": decoy_srcs,
                    "clean_premise_available": True,
                })
    if not reference_ok(score_item(item.reference, item.as_score_input())):
        return None
    return item


def build_defence_item(level: int, seed: int, ordering: str,
                       profile: str = "FULL") -> Optional[Item]:
    n = max(2, min(5, 2 + level // 4))
    n_strict = 1 if level >= 7 else 0
    n_decoy = 0 if level < 6 else min(2, 1 + (level - 6) // 5)
    it = iter(_names(stable_seed(seed, level, ordering, "def"), _POOL))
    sup = max(2, min(5, 2 + level // 4))
    atk_d = max(2, min(5, 2 + level // 4))
    n_junc = junctions_for(level, max(1, n * 4))
    d = build_defence(n, ordering, it, support_depth=sup, junction=(level >= 6),
                      n_junctions=n_junc,
                      ternary=wants_ternary(level, 0),
                      n_strict_attackers=n_strict, n_decoys=n_decoy,
                      attacker_depth=atk_d)
    if d is None:
        return None
    extra: List[Operation] = []
    n_ds = 0 if level < 4 else min(1 + (level - 4) // 4, 3)
    for j in range(n_ds):
        a, b2, c2 = next(it), next(it), next(it)
        extra.append(Operation(kind="premise", content=a))
        extra.append(Operation(kind="defeasible", name=f"sd{j}a", antecedents=(a,), consequent=b2))
        extra.append(Operation(kind="strict", name=f"sd{j}b", antecedents=(b2,), consequent=c2))
    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    extra = list(extra) + _lx
    _allops, _rmap = randomize_rule_names(list(d.all_ops()) + extra,
                                          stable_seed(seed, level, ordering, "drn"))
    base = _ops_ordered(_allops, shuffle_seed=stable_seed(seed, level, ordering, "dshuf"))
    atoms, rnames = _atoms_and_rules(base)
    if atoms & rnames:
        return None
    n_moves_est = len(d.attackers) * (atk_d + 1)
    if n_moves_est <= 14:
        mn = verify_defence_minimum(d)
        if mn.get("witness_moves") is None:
            mn = verify_defence_decomposed(d)
    else:
        mn = verify_defence_decomposed(d)
    if mn.get("witness_moves") is None:
        return None
    src = None
    for o in d.ops:
        if o.kind == "premise":
            src = o.content
            break
    ref_lines = []
    for i, a in enumerate(d.attackers):
        cut = (a.rules[0] if a.rules else a.name)
        ref_lines.append(f"[defeasible z{i}_0: {src} => -{cut}]")
    ref = remap_text("[answer]\n" + "\n".join(ref_lines) + "\n[/answer]", _rmap)
    goals = [{"claim": d.target, "current": d.status(), "want": "JUSTIFIED"}]
    item = Item(task=TASK, level=level, ordering=ordering, mode=DEFENCE,
                prompt=render(_render_ops(base), ordering, goals),
                theory_text=_render_ops(base), base_ops=base, goals=goals,
                reference=ref, min_directives=mn["witness_moves"],
                metadata={
                    "n_attackers": n, "n_strict_attackers": n_strict, "n_decoys": n_decoy,
                    "n_junctions": n_junc,
                    "support_depth": sup, "attacker_depth": atk_d,
                    "n_decoy_strict": n_ds, "shuffled_presentation": True, "profile": profile,
                    "minimality_proven": mn.get("lower_bound_proven", mn.get("proven", True)),
                    "minimality_method": mn.get("method", "per_attacker_lower_bound"),
                    "min_moves": mn["witness_moves"],
                    "lower_bound_proven": mn.get("lower_bound_is_n"),
                    "searched_exhaustively": mn.get("searched_exhaustively"),
                    "minimum_method": mn.get("method", "joint"),
                    "base_status": d.status(),
                })
    if not reference_ok(score_item(item.reference, item.as_score_input())):
        return None
    return item


def build_mixed_item(level: int, seed: int, ordering: str,
                     profile: str = "FULL") -> Optional[Item]:
    n_atk = max(2, min(4, 2 + level // 5))
    it = iter(_names(stable_seed(seed, level, ordering, "mix"), _POOL))
    depth = max(2, min(5, 2 + level // 4))
    stem = max(1, min(4, 1 + level // 5))
    n_junc = junctions_for(level, max(1, n_atk * 4))
    m = build_mixed(it, ordering, n_attackers=n_atk, extra_attack_routes=1, shared=True,
                    depth=depth, shared_depth=stem, junction=(level >= 6),
                    n_junctions=n_junc,
                    ternary=wants_ternary(level, 0))
    if m is None:
        return None
    inter = check_interference(m)
    if not inter["interferes"]:
        return None
    sol = solve_mixed(m)
    if not sol["achieved"] or not sol["all_necessary"]:
        return None
    extra: List[Operation] = []
    n_ds = 0 if level < 4 else min(1 + (level - 4) // 4, 3)
    for j in range(n_ds):
        a, b2, c2 = next(it), next(it), next(it)
        extra.append(Operation(kind="premise", content=a))
        extra.append(Operation(kind="defeasible", name=f"ms{j}a", antecedents=(a,), consequent=b2))
        extra.append(Operation(kind="strict", name=f"ms{j}b", antecedents=(b2,), consequent=c2))
    _lx, _ = language_enrichment(it, [900], prefix="lx")
    _lx = PROFILES[profile].filter(_lx)
    extra = list(extra) + _lx
    _allops, _rmap = randomize_rule_names(list(m.all_ops()) + extra,
                                          stable_seed(seed, level, ordering, "mrn"))
    base = _ops_ordered(_allops, shuffle_seed=stable_seed(seed, level, ordering, "mshuf"))
    atoms, rnames = _atoms_and_rules(base)
    if atoms & rnames:
        return None
    src = None
    for o in m.ops:
        if o.kind == "premise":
            src = o.content
            break
    own = m.attack_own_rules[-1]
    ref = "[answer]\n" + "\n".join(
        [f"[defeasible sa: {src} => -{own}]"]
        + [f"[defeasible sd{j}: {src} => -{x}]" for j, x in enumerate(m.attacker_rules)]
    )
    ref = remap_text(ref + "\n[/answer]", _rmap)
    goals = [{"claim": m.attack_target, "current": "JUSTIFIED", "want": "OVERRULED"},
             {"claim": m.defence_target, "current": m.status(m.defence_target),
              "want": "JUSTIFIED"}]
    item = Item(task=TASK, level=level, ordering=ordering, mode=MIXED,
                prompt=render(_render_ops(base), ordering, goals),
                theory_text=_render_ops(base), base_ops=base, goals=goals,
                reference=ref, min_directives=sol["n_directives"],
                metadata={
                    "minimality_proven": sol.get("all_necessary", False),
                    "minimality_method": "interaction_all_necessary",
                    "n_attackers": n_atk, "n_junctions": n_junc, "shared_node": m.shared_node,
                    "branch_depth": depth, "stem_depth": stem,
                    "n_decoy_strict": n_ds, "shuffled_presentation": True, "profile": profile,
                    "minimality_proven": sol.get("all_necessary", False),
                    "minimality_method": "interaction_all_necessary",
                    "interferes": inter["interferes"],
                    "naive_attack_sabotages_defence":
                        inter.get("naive_B") == "OVERRULED",
                    "min_moves": sol["n_directives"],
                    "all_reference_directives_necessary": sol["all_necessary"],
                })
    if not reference_ok(score_item(item.reference, item.as_score_input())):
        return None
    return item


def reference_ok(result: Dict) -> bool:
    return result["score"] >= 0.999 and not result["diagnostics"]["illegal"]


MODE_BUILDERS = {ATTACK: build_attack_item, DEFENCE: build_defence_item, MIXED: build_mixed_item}


def make_item(level: int, seed: int, ordering: str = LAST_LINK,
              mode: Optional[str] = None, profile: str = "FULL",
              tries: int = 12) -> Optional[Item]:
    if mode is None:
        mode = spec_for(level, ordering, variant=seed % 5).mode
    fn = MODE_BUILDERS.get(mode)
    if fn is None:
        return None
    for k in range(tries):
        it = fn(level, seed * 31 + k, ordering, profile)
        if it is not None:
            return it
    return None
