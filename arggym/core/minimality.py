from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier


@dataclass(frozen=True)
class Move:
    label: str
    chain: int
    ops: Tuple[Operation, ...]

    def size(self) -> int:
        return len(self.ops)


def achieves(base_ops: Sequence[Operation], moves: Sequence[Move], claim: str,
             want: str, ordering: str) -> bool:
    added: List[Operation] = []
    for m in moves:
        added.extend(m.ops)
    plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
    prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
    try:
        v = ASPICVerifier.from_operations(list(base_ops) + plain + prefs, ordering=ordering)
        return str(v.status(claim)) == want and v.is_consistent()
    except Exception:
        return False


def enumerate_moves(chains, target: str, src: str, ordering: str) -> List[Move]:
    out: List[Move] = []
    for ci, ch in enumerate(chains):
        for r in ch.rules:
            if r.get("strict"):
                continue
            out.append(Move(f"undercut:{r['name']}", ci,
                            (Operation(kind="defeasible", name=f"u_{ci}_{r['name']}",
                                       antecedents=(src,), consequent="-" + r["name"]),)))
        if not ch.root_is_axiom:
            out.append(Move(f"undermine:{ch.root}", ci,
                            (Operation(kind="premise", content="-" + ch.root),
                             Operation(kind="prefer_premise", stronger="-" + ch.root,
                                       weaker=ch.root))))
    out.append(Move("rebut:target", -1,
                    (Operation(kind="defeasible", name="w_t", antecedents=(src,),
                               consequent="-" + target),)))
    return out


def lower_bound_is_n(base_ops, chains, target, src, want, ordering) -> Tuple[bool, List[int]]:
    moves = enumerate_moves(chains, target, src, ordering)
    skippable: List[int] = []
    for ci in range(len(chains)):
        others = [m for m in moves if m.chain != ci]
        if achieves(base_ops, others, target, want, ordering):
            skippable.append(ci)
    return (not skippable), skippable


def find_minimum(base_ops, chains, target, src, want, ordering,
                 max_size: Optional[int] = None,
                 exhaustive_limit: int = 20000) -> Dict:
    moves = enumerate_moves(chains, target, src, ordering)
    n_chains = len(chains)
    cap = max_size or (n_chains + 2)

    witness: Optional[List[Move]] = None
    checked = 0

    lb_proven, lb_skippable = lower_bound_is_n(base_ops, chains, target, src, want, ordering)
    floor = n_chains if lb_proven else 1

    forced: List[Move] = []
    if moves and achieves(base_ops, moves, target, want, ordering):
        for i, m in enumerate(moves):
            checked += 1
            rest = [x for j, x in enumerate(moves) if j != i]
            if not (rest and achieves(base_ops, rest, target, want, ordering)):
                forced.append(m)
        if forced:
            checked += 1
            if achieves(base_ops, forced, target, want, ordering):
                witness = list(forced)

    if witness is None:
        optional = [m for m in moves if m not in forced]
        start = max(0, floor - len(forced))
        for extra in range(start, len(optional) + 1):
            if len(forced) + extra > cap:
                break
            hit = None
            for combo in itertools.combinations(optional, extra):
                checked += 1
                if checked > exhaustive_limit:
                    break
                cand = forced + list(combo)
                if achieves(base_ops, cand, target, want, ordering):
                    hit = cand
                    break
            if hit is not None:
                witness = hit
                break
            if checked > exhaustive_limit:
                break

    proven, skippable = lower_bound_is_n(base_ops, chains, target, src, want, ordering)
    n_directives = sum(m.size() for m in witness) if witness else None
    return {
        "found": witness is not None,
        "n_chains": n_chains,
        "n_moves_available": len(moves),
        "required_moves": len(witness) if witness else None,
        "rejected_moves": (len(moves) - len(witness)) if witness else None,
        "witness": [m.label for m in witness] if witness else None,
        "witness_moves": len(witness) if witness else None,
        "witness_directives": n_directives,
        "lower_bound_is_n_chains": proven,
        "skippable_chains": skippable,
        "searched_exhaustively": checked <= exhaustive_limit,
        "combos_checked": checked,
        "forced_moves": len(forced),
        "minimality_proven": witness is not None and checked <= exhaustive_limit,
        "search_floor": floor,
    }


def find_minimum_decomposed(base_ops, chains, target, src, want, ordering) -> Dict:
    moves = enumerate_moves(chains, target, src, ordering)
    n = len(chains)

    target_moves = [m for m in moves if m.chain == -1]
    for tm in target_moves:
        if achieves(base_ops, [tm], target, want, ordering):
            return {"found": False, "reason": "target_attackable_alone",
                    "n_chains": n, "n_moves_available": len(moves)}

    per_chain: List[Optional[Move]] = [None] * n
    for ci in range(n):
        cands = sorted([m for m in moves if m.chain == ci], key=lambda m: m.size())
        if not cands:
            return {"found": False, "reason": f"chain_{ci}_has_no_moves",
                    "n_chains": n, "n_moves_available": len(moves)}
        per_chain[ci] = cands[0]

    for ci in range(n):
        others = [per_chain[j] for j in range(n) if j != ci]
        if achieves(base_ops, others, target, want, ordering):
            return {"found": False, "reason": f"chain_{ci}_skippable",
                    "n_chains": n, "n_moves_available": len(moves)}

    witness = [m for m in per_chain if m is not None]
    if not achieves(base_ops, witness, target, want, ordering):
        for ci in range(n):
            cands = sorted([m for m in moves if m.chain == ci], key=lambda m: m.size())
            for alt in cands[1:]:
                trial = list(witness)
                trial[ci] = alt
                if achieves(base_ops, trial, target, want, ordering):
                    witness = trial
                    break
        if not achieves(base_ops, witness, target, want, ordering):
            return {"found": False, "reason": "no_one_move_per_chain_solution",
                    "n_chains": n, "n_moves_available": len(moves)}

    proven, skippable = lower_bound_is_n(base_ops, chains, target, src, want, ordering)
    directives = sum(m.size() for m in witness)
    return {
        "minimality_proven": proven,
        "minimality_method": "chain_decomposition_with_lower_bound",
        "combos_checked": 0,
        "found": True,
        "n_chains": n,
        "n_moves_available": len(moves),
        "witness": [m.label for m in witness],
        "witness_moves": len(witness),
        "witness_directives": directives,
        "required_moves": len(witness),
        "rejected_moves": len(moves) - len(witness),
        "lower_bound_is_n_chains": proven,
        "skippable_chains": skippable,
        "searched_exhaustively": False,
        "method": "decomposed",
        "combos_checked": 0,
    }
