from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.aspic.api import ASPICVerifier

LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"


@dataclass
class MixedItem:
    attack_target: str
    defence_target: str
    shared_node: Optional[str]
    shared_rule: Optional[str]
    attack_own_rules: List[str]
    defence_own_rules: List[str]
    attacker_rules: List[str]
    ops: List[Operation] = field(default_factory=list)
    ordering: str = LAST_LINK
    provenance: List[str] = field(default_factory=list)

    def all_ops(self) -> List[Operation]:
        facts = [o for o in self.ops if o.kind in ("premise", "axiom")]
        rules = [o for o in self.ops if o.kind in ("defeasible", "strict")]
        prefs = [o for o in self.ops if o.kind in ("prefer_rule", "prefer_premise")]
        return facts + rules + prefs

    def status(self, lit: str, extra: Sequence[Operation] = ()) -> str:
        added = list(extra)
        plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
        prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
        try:
            v = ASPICVerifier.from_operations(self.all_ops() + plain + prefs,
                                              ordering=self.ordering)
            return str(v.status(lit))
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


def build_mixed(names: Sequence[str], ordering: str = LAST_LINK,
                n_attackers: int = 2, extra_attack_routes: int = 1,
                shared: bool = True, depth: int = 2,
                shared_depth: int = 1, junction: bool = False,
                n_junctions: int = 1, ternary: bool = False) -> Optional[MixedItem]:
    it = iter(names)
    root = next(it)
    shared_node = next(it)
    tA = next(it)
    tB = next(it)

    ops: List[Operation] = [Operation(kind="premise", content=root)]
    cur = root
    shared_rule = None
    j_points = set()
    if junction and max(1, shared_depth) >= 1:
        for k in range(max(1, n_junctions)):
            j_points.add(k % max(1, shared_depth))
    junction_at = -1
    for i in range(max(1, shared_depth)):
        nxt = shared_node if i == max(1, shared_depth) - 1 else next(it)
        nm = f"k{i}"
        if i in j_points:
            extra = []
            for _e in range(2 if ternary else 1):
                broot = next(it); blit = next(it)
                ops.append(Operation(kind="premise", content=broot))
                ops.append(Operation(kind="defeasible", name=f"{nm}b{_e}",
                                     antecedents=(broot,), consequent=blit))
                extra.append(blit)
            ops.append(Operation(kind="defeasible", name=nm,
                                 antecedents=tuple([cur] + extra), consequent=nxt))
        else:
            ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
        shared_rule = nm
        cur = nxt

    a_rules: List[str] = []
    a_src = shared_node if shared else next(it)
    if not shared:
        ops.append(Operation(kind="premise", content=a_src))
    for i in range(max(1, extra_attack_routes)):
        cur = a_src
        for j in range(max(2, depth)):
            last = j == max(2, depth) - 1
            nxt = tA if last else next(it)
            nm = f"a{i}{j}"
            ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
            a_rules.append(nm)
            cur = nxt

    b_src = shared_node if shared else next(it)
    if not shared:
        ops.append(Operation(kind="premise", content=b_src))
    b_rules = []
    cur = b_src
    for j in range(max(2, depth)):
        last = j == max(2, depth) - 1
        nxt = tB if last else next(it)
        nm = f"b{j}"
        ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
        b_rules.append(nm)
        cur = nxt

    atk: List[str] = []
    for j in range(max(1, n_attackers)):
        ar = next(it)
        ops.append(Operation(kind="premise", content=ar))
        cur = ar
        for q in range(max(1, depth - 1)):
            last = q == max(1, depth - 1) - 1
            nxt = "-" + tB if last else next(it)
            nm = f"x{j}" if last else f"x{j}_{q}"
            ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
            cur = nxt
        atk.append(f"x{j}")

    item = MixedItem(attack_target=tA, defence_target=tB,
                     shared_node=shared_node if shared else None,
                     shared_rule=shared_rule if shared else None,
                     attack_own_rules=a_rules, defence_own_rules=b_rules,
                     attacker_rules=atk, ops=ops, ordering=ordering)

    if item.status(tA) != "JUSTIFIED":
        return None
    if item.status(tB) == "JUSTIFIED":
        return None
    item.provenance.append(f"shared={shared} attack_routes={extra_attack_routes} "
                           f"attackers={n_attackers} base A={item.status(tA)} B={item.status(tB)}")
    return item


def check_interference(item: MixedItem) -> Dict:
    src = None
    for o in item.ops:
        if o.kind == "premise":
            src = o.content
            break
    out: Dict = {"shared": item.shared_node is not None}

    def undercut(rule: str) -> List[Operation]:
        return [Operation(kind="defeasible", name=f"zz_{rule}", antecedents=(src,),
                          consequent="-" + rule)]

    if item.shared_rule:
        naive = undercut(item.shared_rule)
        out["naive_A"] = item.status(item.attack_target, naive)
        out["naive_B"] = item.status(item.defence_target, naive)
    own = item.attack_own_rules[-1] if item.attack_own_rules else None
    if own:
        surg = undercut(own)
        out["surgical_A"] = item.status(item.attack_target, surg)
        out["surgical_B"] = item.status(item.defence_target, surg)
    out["interferes"] = (out.get("naive_A") == "OVERRULED"
                         and out.get("naive_B") == "OVERRULED"
                         and out.get("surgical_A") == "OVERRULED"
                         and out.get("surgical_B") != "OVERRULED")
    return out


def solve_mixed(item: MixedItem) -> Dict:
    src = None
    for o in item.ops:
        if o.kind == "premise":
            src = o.content
            break
    own = item.attack_own_rules[-1]
    moves: List[Tuple[str, List[Operation]]] = [
        (f"undercut:{own}", [Operation(kind="defeasible", name="sa", antecedents=(src,),
                                       consequent="-" + own)])]
    for j, x in enumerate(item.attacker_rules):
        moves.append((f"undercut:{x}", [Operation(kind="defeasible", name=f"sd{j}",
                                                  antecedents=(src,), consequent="-" + x)]))

    def ok(sel) -> bool:
        added: List[Operation] = []
        for _l, ops in sel:
            added.extend(ops)
        return (item.status(item.attack_target, added) == "OVERRULED"
                and item.status(item.defence_target, added) == "JUSTIFIED")

    achieved = ok(moves)
    necessary = []
    for m in moves:
        rest = [x for x in moves if x is not m]
        necessary.append(not ok(rest))
    return {
        "achieved": achieved,
        "n_directives": sum(len(ops) for _l, ops in moves),
        "moves": [l for l, _ in moves],
        "all_necessary": all(necessary) if achieved else None,
    }
