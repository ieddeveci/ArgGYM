from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.core.curriculum import negated_branch


def _is_last(ordering: str) -> bool:
    return str(ordering).startswith("last_link")


def _is_weakest(ordering: str) -> bool:
    return str(ordering).startswith("weakest_link")


LAST_LINK = "last_link_elitist"
WEAKEST_LINK = "weakest_link_elitist"

UNDERMINE = "undermine_root"
UNDERCUT_FINAL = "undercut_final"
UNDERCUT_MID = "undercut_mid"
REBUT = "rebut_target"
FLIP_PREF = "flip_preference"
ALL_ATTACKS = (UNDERMINE, UNDERCUT_FINAL, UNDERCUT_MID, REBUT, FLIP_PREF)


@dataclass
class Chain:
    root: str
    root_is_axiom: bool
    mids: List[str]
    target: str
    rules: List[dict]
    extra_ops: List[Operation] = field(default_factory=list)
    config: str = ""

    def rule_names(self) -> List[str]:
        return [r["name"] for r in self.rules]

    def defeasible_rules(self) -> List[dict]:
        return [r for r in self.rules if not r.get("strict")]

    def final_rule(self) -> dict:
        return self.rules[-1]

    def to_ops(self) -> List[Operation]:
        ops = [Operation(kind="axiom" if self.root_is_axiom else "premise", content=self.root)]
        for r in self.rules:
            ops.append(Operation(kind="strict" if r.get("strict") else "defeasible",
                                 name=r["name"], antecedents=tuple(r["ants"]),
                                 consequent=r["cons"]))
        plain = [o for o in self.extra_ops if o.kind not in ("prefer_rule", "prefer_premise")]
        prefs = [o for o in self.extra_ops if o.kind in ("prefer_rule", "prefer_premise")]
        return ops + plain + prefs


def _mk(root, root_axiom, mids, target, specs, extra=None, config="") -> Chain:
    rules = []
    seq = [root] + list(mids)
    for i, (name, strict) in enumerate(specs):
        rules.append({"name": name, "ants": [seq[i]], "cons": (seq + [target])[i + 1],
                      "strict": strict})
    return Chain(root=root, root_is_axiom=root_axiom, mids=list(mids), target=target,
                 rules=rules, extra_ops=list(extra or []), config=config)


def _chain_of(root: str, root_axiom: bool, target: str, depth: int,
              rule_names: Sequence[str], strict_at: Sequence[int], config: str,
              mid_names: Sequence[str], n_junctions: int = 0,
              ternary: bool = False) -> Chain:
    assert depth >= 1
    mids = list(mid_names[:max(0, depth - 1)])
    seq = [root] + mids + [target]
    strict_set = set(strict_at)
    rules = []
    for i in range(depth):
        rules.append({"name": rule_names[i], "ants": [seq[i]], "cons": seq[i + 1],
                      "strict": i in strict_set})

    extra: List[Operation] = []
    if n_junctions > 0:
        eligible = [i for i in range(depth) if i not in strict_set]
        step = max(1, len(eligible) // (n_junctions + 1)) if eligible else 1
        picks = [eligible[min(len(eligible) - 1, step * (k + 1) - 1)]
                 for k in range(n_junctions)] if eligible else []
        for k, i in enumerate(sorted(set(picks))):
            n_extra = 2 if (ternary and k % 2 == 0) else 1
            for e in range(n_extra):
                broot = f"{root}_jr{i}{e}"
                if negated_branch(i * 2 + e):
                    broot = "-" + broot
                blit = f"{root}_jl{i}{e}"
                brule = f"{rule_names[i]}j{i}{e}"
                extra.append(Operation(kind="premise", content=broot))
                extra.append(Operation(kind="defeasible", name=brule,
                                       antecedents=(broot,), consequent=blit))
                rules[i]["ants"].append(blit)

    return Chain(root=root, root_is_axiom=root_axiom, mids=mids, target=target,
                 rules=rules, extra_ops=extra, config=config)


def build_c1(root, mids, target, rn, depth=2,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    return _chain_of(root, False, target, depth, rn, (), "C1", mids, n_junctions=n_junctions, ternary=ternary)


def build_c2(root, mids, target, rn, depth=2,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    d = max(2, depth)
    return _chain_of(root, False, target, d, rn, (d - 1,), "C2", mids,
                     n_junctions=n_junctions, ternary=ternary)


def build_c3(root, mids, target, rn, depth=2,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    return _chain_of(root, True, target, depth, rn, (), "C3", mids, n_junctions=n_junctions, ternary=ternary)


def build_c4(root, mids, target, rn, depth=2,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    d = max(2, depth)
    return _chain_of(root, True, target, d, rn, (d - 1,), "C4", mids, n_junctions=n_junctions, ternary=ternary)


def build_c5(root, mids, target, rn, rival_premise, rival_rule,
             ordering: str = LAST_LINK, depth: int = 2) -> Chain:
    ch = _chain_of(root, False, target, depth, rn, (), "C5", mids)
    extras = [
        Operation(kind="premise", content=rival_premise),
        Operation(kind="defeasible", name=rival_rule, antecedents=(rival_premise,),
                  consequent="-" + target),
    ]
    if _is_last(ordering):
        extras.append(Operation(kind="prefer_rule",
                                stronger=ch.rules[-1]["name"], weaker=rival_rule))
    else:
        for r in ch.rules:
            extras.append(Operation(kind="prefer_rule", stronger=r["name"], weaker=rival_rule))
        extras.append(Operation(kind="prefer_premise", stronger=root, weaker=rival_premise))
    ch.extra_ops = extras
    return ch


def build_c7(root, mids, target, rn, depth=4, n_defeasible=2,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    d = max(2, depth)
    k = max(1, min(n_defeasible, d))
    step = d / k
    pos = sorted({min(d - 1, int(i * step)) for i in range(k)})
    strict_at = tuple(i for i in range(d) if i not in set(pos))
    ch = _chain_of(root, False, target, d, rn, strict_at, "C7", mids, n_junctions=n_junctions, ternary=ternary)
    return ch


def build_c9(root, mids, target, rn, depth=3,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    d = max(3, depth)
    mid_strict = (d // 2,) if d >= 3 else ()
    return _chain_of(root, False, target, d, rn, mid_strict, "C9", mids, n_junctions=n_junctions, ternary=ternary)


def build_c8(root, mids, target, rn, depth=3,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    d = max(3, depth)
    ch = _chain_of(root, False, target, d, rn, (d - 1,), "C8", mids, n_junctions=n_junctions, ternary=ternary)
    broot = f"{root}_b"
    blit = f"{mids[0]}_b" if mids else f"{target}_b"
    brule = f"{rn[0]}b"
    at = min(d // 2, d - 2)
    # Overwrites rather than appends, so a budgeted junction already at `at` would lose its
    # branch literal and leave an orphan premise. There is no depth-free threshold that makes
    # this safe: over depth 2-9 x n_junctions 0-2, the pairs (2,2), (3,2) and (7,2) collide.
    # What holds is narrower -- `build_attack_item` forms none of those pairs. It divides
    # `j_alloc` into a `per_chain` of 0, 1 or 2, and the depths it pairs those with over
    # levels 1-30 and all four profiles never put a pick on `at`. A caller choosing its own
    # depth and `n_junctions` has to check the pair.
    ch.rules[at]["ants"] = [ch.rules[at]["ants"][0], blit]
    ch.extra_ops.extend([
        Operation(kind="premise", content=broot),
        Operation(kind="defeasible", name=brule, antecedents=(broot,), consequent=blit),
    ])
    return ch


def build_c6(root, mids, target, rn, depth=3,
             n_junctions: int = 0, ternary: bool = False) -> Chain:
    d = max(2, depth)
    return _chain_of(root, False, target, d, rn, tuple(range(1, d)), "C6", mids, n_junctions=n_junctions, ternary=ternary)


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    builder: Callable
    n_mids: int
    n_rules: int
    profile: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]
    note: str
    #: Junctions the config builds whatever `n_junctions` it is asked for. C8 is the one:
    #: two branches feeding one step is what C8 *is*, so its junction arrives before any
    #: budget is divided. A caller reading a junction budget as a ceiling has to add these,
    #: or the ceiling sits below what the picks already brought (#97).
    structural_junctions: int = 0

    @staticmethod
    def _axis(ordering: str) -> str:
        return ("weakest_link_elitist" if str(ordering).startswith("weakest_link")
                else "last_link_elitist")

    def admits(self, ordering: str) -> FrozenSet[str]:
        return self.profile[self._axis(ordering)][0]

    def blocks(self, ordering: str) -> FrozenSet[str]:
        return self.profile[self._axis(ordering)][1]


_ALL4 = frozenset({UNDERMINE, UNDERCUT_FINAL, UNDERCUT_MID, REBUT})

CONFIGS: Dict[str, ConfigSpec] = {
    "C1": ConfigSpec("C1", build_c1, 1, 2,
                     {LAST_LINK:    (_ALL4, frozenset({FLIP_PREF})),
                      WEAKEST_LINK: (_ALL4, frozenset({FLIP_PREF}))},
                     "all defeasible, ordinary root: every attack available"),

    "C2": ConfigSpec("C2", build_c2, 1, 2,
                     {LAST_LINK:    (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({UNDERCUT_FINAL, REBUT, FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({UNDERCUT_FINAL, REBUT, FLIP_PREF}))},
                     "strict final rule: cannot undercut it, cannot rebut the target"),

    "C3": ConfigSpec("C3", build_c3, 1, 2,
                     {LAST_LINK:    (frozenset({UNDERCUT_FINAL, UNDERCUT_MID, REBUT}),
                                     frozenset({UNDERMINE, FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERCUT_FINAL, UNDERCUT_MID}),
                                     frozenset({UNDERMINE, REBUT, FLIP_PREF}))},
                     "axiom root: no undermining; and no rebutting under weakest-link"),

    "C4": ConfigSpec("C4", build_c4, 1, 2,
                     {LAST_LINK:    (frozenset({UNDERCUT_MID}),
                                     frozenset({UNDERMINE, UNDERCUT_FINAL, REBUT, FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERCUT_MID}),
                                     frozenset({UNDERMINE, UNDERCUT_FINAL, REBUT, FLIP_PREF}))},
                     "axiom root AND strict final: only the middle defeasible rule"),

    "C5": ConfigSpec("C5", build_c5, 1, 2,
                     {LAST_LINK:    (frozenset({UNDERMINE, UNDERCUT_FINAL, UNDERCUT_MID}),
                                     frozenset({FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERMINE, UNDERCUT_FINAL, UNDERCUT_MID}),
                                     frozenset({FLIP_PREF}))},
                     "existing rival + preference: a DECOY, the preference is not flippable"),

    "C7": ConfigSpec("C7", build_c7, 3, 4,
                     {LAST_LINK:    (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({REBUT, FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({REBUT, FLIP_PREF}))},
                     "scarce: k defeasible rules scattered among d, interpolating C1..C6"),

    "C9": ConfigSpec("C9", build_c9, 2, 3,
                     {LAST_LINK:    (frozenset({UNDERMINE, UNDERCUT_MID, REBUT}),
                                     frozenset({FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERMINE, UNDERCUT_MID, REBUT}),
                                     frozenset({FLIP_PREF}))},
                     "defeasible final rule, so the target is rebuttable here; a mid-chain strict "
                     "rule keeps the cut point a choice rather than a formula"),

    "C8": ConfigSpec("C8", build_c8, 2, 3,
                     {LAST_LINK:    (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({REBUT, FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({REBUT, FLIP_PREF}))},
                     "junction: two branches feed one step, so the chain has TWO roots to undermine "
                     "and an extra rule to undercut; rebut blocked by the strict final rule",
                     structural_junctions=1),

    "C6": ConfigSpec("C6", build_c6, 2, 3,
                     {LAST_LINK:    (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({UNDERCUT_FINAL, REBUT, FLIP_PREF})),
                      WEAKEST_LINK: (frozenset({UNDERMINE, UNDERCUT_MID}),
                                     frozenset({UNDERCUT_FINAL, REBUT, FLIP_PREF}))},
                     "one defeasible rule beneath two strict: must search past both"),
}


def attack_ops(chain: Chain, kind: str, src: str, ordering: str,
               idx: int = 0) -> Optional[List[Operation]]:
    tgt = chain.target
    dfs = chain.defeasible_rules()
    if kind == UNDERMINE:
        if chain.root_is_axiom:
            return None
        return [Operation(kind="premise", content="-" + chain.root),
                Operation(kind="prefer_premise", stronger="-" + chain.root, weaker=chain.root)]
    if kind == UNDERCUT_FINAL:
        fr = chain.final_rule()
        if fr.get("strict"):
            return None
        return [Operation(kind="defeasible", name=f"ku{idx}", antecedents=(src,),
                          consequent="-" + fr["name"])]
    if kind == UNDERCUT_MID:
        cand = [r for r in dfs if r["name"] != chain.final_rule()["name"]] or dfs
        if not cand:
            return None
        return [Operation(kind="defeasible", name=f"km{idx}", antecedents=(src,),
                          consequent="-" + cand[0]["name"])]
    if kind == REBUT:
        nm = f"kr{idx}"
        ops = [Operation(kind="defeasible", name=nm, antecedents=(src,), consequent="-" + tgt)]
        if _is_last(ordering):
            fr = chain.final_rule()
            if fr.get("strict"):
                return None
            ops.append(Operation(kind="prefer_rule", stronger=nm, weaker=fr["name"]))
        else:
            if any(r.get("strict") for r in chain.rules):
                return None
            for r in dfs:
                ops.append(Operation(kind="prefer_rule", stronger=nm, weaker=r["name"]))
            if chain.root_is_axiom:
                return None
            ops.append(Operation(kind="prefer_premise", stronger=src, weaker=chain.root))
        return ops
    if kind == FLIP_PREF:
        pref = next((o for o in chain.extra_ops if o.kind == "prefer_rule"), None)
        if pref is None:
            return None
        return [Operation(kind="prefer_rule", stronger=pref.weaker, weaker=pref.stronger)]
    return None


def verify_config(spec: ConfigSpec, ordering: str = LAST_LINK,
                  depth: int = 2) -> Tuple[bool, Dict[str, str], str]:
    root, tgt = "aa1", "zz1"
    mids = [f"bb{i}" for i in range(1, depth + 2)]
    rn = [f"p{i}" for i in range(1, depth + 2)]
    if spec.name == "C5":
        chain = spec.builder(root, mids, tgt, rn, "cc1", "q9", ordering, depth)
    else:
        chain = spec.builder(root, mids, tgt, rn, depth)
    base = chain.to_ops() + [Operation(kind="premise", content="src")]
    try:
        if str(ASPICVerifier.from_operations(base, ordering=ordering).status(tgt)) != "JUSTIFIED":
            return False, {}, "target not JUSTIFIED in the base chain"
    except Exception as e:
        return False, {}, f"base chain failed to build: {type(e).__name__}: {e}"

    results: Dict[str, str] = {}
    for kind in ALL_ATTACKS:
        ops = attack_ops(chain, kind, "src", ordering, idx=1)
        if ops is None:
            results[kind] = "not-expressible"
            continue
        plain = [o for o in ops if o.kind not in ("prefer_rule", "prefer_premise")]
        prefs = [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
        try:
            v = ASPICVerifier.from_operations(base + plain + prefs, ordering=ordering)
            results[kind] = str(v.status(tgt))
        except Exception as e:
            results[kind] = f"ERR:{type(e).__name__}"

    got_admit = {k for k, v in results.items() if v == "OVERRULED"}
    declared_admit = set(spec.admits(ordering))
    if FLIP_PREF not in declared_admit:
        got_admit.discard(FLIP_PREF)
    wrong_admit = declared_admit - got_admit
    wrong_block = set(spec.blocks(ordering)) & got_admit
    ok = not wrong_admit and not wrong_block
    detail = ""
    if wrong_admit:
        detail += f"declared-admits that did NOT reach OVERRULED: {sorted(wrong_admit)}; "
    if wrong_block:
        detail += f"declared-blocks that DID reach OVERRULED: {sorted(wrong_block)}"
    return ok, results, detail


def reasoning_cost(chain: Chain, ordering: str) -> Dict[str, float]:
    base = chain.to_ops() + [Operation(kind="premise", content="src")]
    tgt = chain.target
    probes: List[Tuple[str, int, bool]] = []

    def try_ops(ops) -> Tuple[bool, int]:
        if ops is None:
            return False, 0
        plain = [o for o in ops if o.kind not in ("prefer_rule", "prefer_premise")]
        prefs = [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
        try:
            v = ASPICVerifier.from_operations(base + plain + prefs, ordering=ordering)
            return str(v.status(tgt)) == "OVERRULED", len(ops)
        except Exception:
            return False, len(ops)

    for r in chain.rules:
        ok, n = try_ops([Operation(kind="defeasible", name="kx", antecedents=("src",),
                                   consequent="-" + r["name"])])
        probes.append((f"undercut:{r['name']}", n, ok))
    if any(o.kind == "prefer_rule" for o in chain.extra_ops):
        ok, n = try_ops(attack_ops(chain, FLIP_PREF, "src", ordering, idx=8))
        probes.append((FLIP_PREF, n, ok))
    for kind in (REBUT, UNDERMINE):
        ops = attack_ops(chain, kind, "src", ordering, idx=9)
        ok, n = try_ops(ops)
        probes.append((kind, n, ok))

    viable = [p for p in probes if p[2]]
    n_viable = len(viable)
    n_blocked = len(probes) - n_viable
    depth = len(chain.rules)
    construction = min((p[1] for p in viable), default=0)
    search = depth / n_viable if n_viable else float(depth)
    return {
        "depth": depth,
        "n_probes": len(probes),
        "n_viable": n_viable,
        "elimination": float(n_blocked),
        "search": round(search, 2),
        "construction": float(construction),
        "cost": round(1.0 * n_blocked + 1.0 * search + 1.0 * construction, 2),
    }


def verify_all(orderings: Sequence[str] = (LAST_LINK, WEAKEST_LINK),
               depths: Sequence[int] = tuple(range(2, 10)), verbose: bool = False) -> bool:
    all_ok = True
    failures: List[str] = []
    for ordering in orderings:
        print(f"\n=== {ordering} ===")
        print(f"  {'config':7}" + "".join(f"{('d'+str(d)):>6}" for d in depths))
        for name in sorted(CONFIGS):
            spec = CONFIGS[name]
            row = ""
            for d in depths:
                ok, results, detail = verify_config(spec, ordering, d)
                row += f"{('ok' if ok else 'FAIL'):>6}"
                if not ok:
                    failures.append(f"{name}/{ordering.split('_')[0]}/d{d}: {detail.strip()}")
                    all_ok = False
                if verbose and not ok:
                    print(f"    {name} d{d}: {results}")
            print(f"  {name:7}{row}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
    return all_ok


if __name__ == "__main__":
    import sys
    print("ArgGym v4 step 1 -- chain configurations verified against the engine")
    ok = verify_all()
    print("\n" + ("all configurations verified" if ok else "SOME CONFIGURATIONS DO NOT MATCH"))
    sys.exit(0 if ok else 1)
