from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier

LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"


@dataclass
class Attacker:
    name: str
    root: str
    mid: Optional[str]
    via_strict: bool
    rules: List[str] = field(default_factory=list)
    ops: List[Operation] = field(default_factory=list)


@dataclass
class DefenceItem:
    target: str
    support_rules: List[str]
    attackers: List[Attacker]
    ops: List[Operation]
    ordering: str
    decoy_ops: List[Operation] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)

    def all_ops(self) -> List[Operation]:
        every = self.ops + self.decoy_ops
        facts = [o for o in every if o.kind in ("premise", "axiom")]
        rules = [o for o in every if o.kind in ("defeasible", "strict")]
        prefs = [o for o in every if o.kind in ("prefer_rule", "prefer_premise")]
        return facts + rules + prefs

    def status(self, lit: Optional[str] = None) -> str:
        try:
            v = ASPICVerifier.from_operations(self.all_ops(), ordering=self.ordering)
            return str(v.status(lit or self.target))
        except Exception:
            return "ERR"

    def render(self) -> str:
        out = []
        for o in self.all_ops():
            if o.kind in ("premise", "axiom"):
                out.append(f"[{o.kind}: {o.content}]")
            elif o.kind in ("defeasible", "strict"):
                arrow = "=>" if o.kind == "defeasible" else "->"
                out.append(f"[{o.kind} {o.name}: {' AND '.join(o.antecedents)} {arrow} {o.consequent}]")
            else:
                out.append(f"[{o.kind}: {o.stronger} > {o.weaker}]")
        return "\n".join(out)


def build_defence(n_attackers: int, ordering: str, names: Sequence[str],
                  support_depth: int = 2, n_strict_attackers: int = 0,
                  n_decoys: int = 0, attacker_depth: int = 2) -> Optional[DefenceItem]:
    it = iter(names)
    tgt = next(it)
    root = next(it)
    ops: List[Operation] = [Operation(kind="premise", content=root)]

    support: List[str] = []
    cur = root
    for i in range(max(1, support_depth)):
        nxt = tgt if i == support_depth - 1 else next(it)
        rn = f"s{i+1}"
        ops.append(Operation(kind="defeasible", name=rn, antecedents=(cur,), consequent=nxt))
        support.append(rn)
        cur = nxt

    attackers: List[Attacker] = []
    ad = max(2, attacker_depth)
    for a in range(n_attackers):
        ar = next(it)
        ops.append(Operation(kind="premise", content=ar))
        strict = a < n_strict_attackers
        cur = ar
        rules: List[str] = []
        first_mid = None
        for q in range(ad):
            last = q == ad - 1
            nxt = "-" + tgt if last else next(it)
            if first_mid is None:
                first_mid = nxt
            nm = f"x{a+1}" if last else f"x{a+1}_{q}"
            ops.append(Operation(kind="strict" if (last and strict) else "defeasible", name=nm,
                                 antecedents=(cur,), consequent=nxt))
            rules.append(nm)
            cur = nxt
        attackers.append(Attacker(name=f"x{a+1}", root=ar, mid=first_mid, via_strict=strict,
                                  rules=rules))

    item = DefenceItem(target=tgt, support_rules=support, attackers=attackers,
                       ops=ops, ordering=ordering)

    for d in range(n_decoys):
        z1, z2, y1, y2 = next(it), next(it), next(it), next(it)
        item.decoy_ops += [
            Operation(kind="premise", content=z1),
            Operation(kind="premise", content=z2),
            Operation(kind="defeasible", name=f"g{d}1", antecedents=(z1,), consequent=y1),
            Operation(kind="defeasible", name=f"g{d}2", antecedents=(y1,), consequent=y2),
            Operation(kind="defeasible", name=f"g{d}3", antecedents=(z2,), consequent="-" + y2),
            Operation(kind="prefer_rule", stronger=f"g{d}2", weaker=f"g{d}3"),
        ]

    base = item.status()
    if base == "JUSTIFIED":
        return None
    item.provenance.append(f"{n_attackers} attackers ({n_strict_attackers} via strict) at depth {ad}, "
                           f"support depth {support_depth}, {n_decoys} decoys, base {base}")
    return item


def defence_moves(item: DefenceItem) -> List[Tuple[str, int, List[Operation]]]:
    out: List[Tuple[str, int, List[Operation]]] = []
    src = item.attackers[0].root if item.attackers else None
    base_root = None
    for o in item.ops:
        if o.kind == "premise":
            base_root = o.content
            break
    for i, a in enumerate(item.attackers):
        for k, rn in enumerate(a.rules or [a.name]):
            if rn == a.name and a.via_strict:
                continue
            out.append((f"undercut:{rn}", i,
                        [Operation(kind="defeasible", name=f"z{i}_{k}",
                                   antecedents=(base_root,), consequent="-" + rn)]))
        if item.ordering == LAST_LINK:
            out.append((f"prefer:{a.name}", i,
                        [Operation(kind="prefer_rule", stronger=item.support_rules[-1],
                                   weaker=a.name)]))
        else:
            out.append((f"prefer-all:{a.name}", i,
                        [Operation(kind="prefer_rule", stronger=s, weaker=a.name)
                         for s in item.support_rules]))
    return out


def verify_minimum(item: DefenceItem, exhaustive_limit: int = 8000) -> Dict:
    import itertools
    moves = defence_moves(item)
    base = item.all_ops()
    n = len(item.attackers)

    def achieves(sel) -> bool:
        added: List[Operation] = []
        for _lbl, _ci, ops in sel:
            added.extend(ops)
        plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
        prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
        try:
            v = ASPICVerifier.from_operations(base + plain + prefs, ordering=item.ordering)
            return str(v.status(item.target)) == "JUSTIFIED" and v.is_consistent()
        except Exception:
            return False

    skippable = []
    for ci in range(n):
        others = [m for m in moves if m[1] != ci]
        if achieves(others):
            skippable.append(ci)

    witness = None
    checked = 0
    for size in range(1, n + 3):
        for combo in itertools.combinations(moves, size):
            checked += 1
            if checked > exhaustive_limit:
                break
            if achieves(combo):
                witness = combo
                break
        if witness or checked > exhaustive_limit:
            break
    return {
        "n_attackers": n,
        "n_moves_available": len(moves),
        "witness_moves": len(witness) if witness else None,
        "witness_directives": sum(len(m[2]) for m in witness) if witness else None,
        "witness": [m[0] for m in witness] if witness else None,
        "lower_bound_is_n": not skippable,
        "skippable": skippable,
        "searched_exhaustively": checked <= exhaustive_limit,
    }


def verify_minimum_decomposed(item: DefenceItem) -> Dict:
    moves = defence_moves(item)
    n = len(item.attackers)
    base = item.all_ops()

    def achieves(sel) -> bool:
        added: List[Operation] = []
        for _lbl, _ci, ops in sel:
            added.extend(ops)
        plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
        prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
        try:
            v = ASPICVerifier.from_operations(base + plain + prefs, ordering=item.ordering)
            return str(v.status(item.target)) == "JUSTIFIED" and v.is_consistent()
        except Exception:
            return False

    per: List[Optional[tuple]] = [None] * n
    for ci in range(n):
        cands = sorted([m for m in moves if m[1] == ci], key=lambda m: len(m[2]))
        if not cands:
            return {"witness_moves": None, "reason": f"attacker_{ci}_has_no_moves"}
        per[ci] = cands[0]

    skippable = []
    for ci in range(n):
        others = [per[j] for j in range(n) if j != ci and per[j] is not None]
        if achieves(others):
            skippable.append(ci)
    if skippable:
        return {"witness_moves": None, "reason": f"attackers_skippable:{skippable}"}

    witness = [m for m in per if m is not None]
    if not achieves(witness):
        for ci in range(n):
            cands = sorted([m for m in moves if m[1] == ci], key=lambda m: len(m[2]))
            for alt in cands[1:]:
                trial = list(witness)
                trial[ci] = alt
                if achieves(trial):
                    witness = trial
                    break
        if not achieves(witness):
            return {"witness_moves": None, "reason": "no_one_move_per_attacker_solution"}

    return {
        "n_attackers": n,
        "n_moves_available": len(moves),
        "witness": [m[0] for m in witness],
        "witness_moves": len(witness),
        "witness_directives": sum(len(m[2]) for m in witness),
        "lower_bound_is_n": True,
        "searched_exhaustively": False,
        "method": "decomposed",
    }
