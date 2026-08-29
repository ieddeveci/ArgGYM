from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def dedupe_parallel(items: Sequence[Tuple[T, str]]) -> Tuple[List[T], List[str]]:
    seen = set()
    objs: List[T] = []
    labels: List[str] = []
    for obj, label in items:
        if label in seen:
            continue
        seen.add(label)
        objs.append(obj)
        labels.append(label)
    return objs, labels


def minimal_subset(candidates: Sequence[T], holds: Callable[[List[T]], bool],
                   max_search: int = 4000) -> Optional[List[T]]:
    full = list(candidates)
    if not holds(full):
        return None
    checked = 0
    for size in range(1, len(full)):
        for combo in itertools.combinations(full, size):
            checked += 1
            if checked > max_search:
                return full
            if holds(list(combo)):
                return list(combo)
    return full


def assert_irredundant(chosen: Sequence[T], holds: Callable[[List[T]], bool]) -> bool:
    full = list(chosen)
    for size in range(1, len(full)):
        for combo in itertools.combinations(full, size):
            if holds(list(combo)):
                return False
    return True


def strategy_candidates(base_ops, goal_claim: str, goal_status: str, ordering: str,
                        seed_lits: Sequence[str], rule_names: Sequence[str],
                        premises: Sequence[str]):
    from arggym.aspic.engine import Operation
    out = []
    tgt = goal_claim.lstrip("-")
    for s in seed_lits:
        out.append(Operation(kind="defeasible", name="cw", antecedents=(s,),
                             consequent=("-" + tgt) if not goal_claim.startswith("-") else tgt))
        for r in rule_names:
            out.append(Operation(kind="defeasible", name=f"cu_{r}", antecedents=(s,),
                                 consequent="-" + r))
    for r in rule_names:
        out.append(Operation(kind="prefer_rule", stronger="cw", weaker=r))
    for p in premises:
        out.append(Operation(kind="premise", content="-" + p))
    return out


def split_atoms_and_rules(ops) -> Tuple[set, set]:
    rules = {o.name for o in ops
             if o.kind in ("defeasible", "strict") and getattr(o, "name", None)}
    atoms = set()
    for o in ops:
        for x in (list(getattr(o, "antecedents", None) or ())
                  + ([o.consequent] if getattr(o, "consequent", None) else [])
                  + ([o.content] if getattr(o, "content", None) else [])):
            bare = x.lstrip("-")
            if bare not in rules:
                atoms.add(bare)
    return atoms, rules


def randomize_rule_names(ops, seed: int, prefix: str = ""):
    import random as _r
    rng = _r.Random(seed)
    old_names = [o.name for o in ops
                 if o.kind in ("defeasible", "strict") and getattr(o, "name", None)]
    pool = [f"{prefix}{a}{b}{c}" for a in "cdfghjklmnpqrstvwxz"
            for b in "aeiouy" for c in "0123456789"]
    rng.shuffle(pool)
    if len(pool) < len(old_names):
        return list(ops), {}
    mapping = {old: pool[i] for i, old in enumerate(old_names)}

    def remap_lit(x):
        if not isinstance(x, str):
            return x
        neg = x.startswith("-")
        bare = x[1:] if neg else x
        if bare in mapping:
            return ("-" if neg else "") + mapping[bare]
        return x

    out = []
    for o in ops:
        if o.kind in ("defeasible", "strict"):
            out.append(type(o)(kind=o.kind, name=mapping.get(o.name, o.name),
                               antecedents=tuple(remap_lit(a) for a in (o.antecedents or ())),
                               consequent=remap_lit(o.consequent)))
        elif o.kind in ("prefer_rule", "prefer_premise"):
            out.append(type(o)(kind=o.kind, stronger=remap_lit(o.stronger),
                               weaker=remap_lit(o.weaker)))
        else:
            out.append(type(o)(kind=o.kind, content=remap_lit(o.content)))
    return out, mapping


def remap_text(text: str, mapping) -> str:
    import re as _re
    if not text or not mapping:
        return text
    for old in sorted(mapping, key=len, reverse=True):
        text = _re.sub(rf"(?<![\w]){_re.escape(old)}(?![\w])", mapping[old], text)
    return text


def negation_coverage(ops) -> dict:
    rules = {o.name for o in ops
             if o.kind in ("defeasible", "strict") and getattr(o, "name", None)}
    out = {"negated_premise": 0, "negated_consequent": 0, "negated_rule_target": 0,
           "negated_antecedent": 0, "negated_pref_operand": 0}
    for o in ops:
        if o.kind in ("premise", "axiom") and (o.content or "").startswith("-"):
            out["negated_premise"] += 1
        if o.kind in ("defeasible", "strict"):
            c = o.consequent or ""
            if c.startswith("-"):
                if c[1:] in rules:
                    out["negated_rule_target"] += 1
                else:
                    out["negated_consequent"] += 1
            for a in (o.antecedents or ()):
                if a.startswith("-"):
                    out["negated_antecedent"] += 1
        if o.kind in ("prefer_rule", "prefer_premise"):
            if (o.stronger or "").startswith("-") or (o.weaker or "").startswith("-"):
                out["negated_pref_operand"] += 1
    return out


def negation_gadget(names_iter, ridx, prefix="ng", winner_feeds=True):
    from arggym.aspic.engine import Operation
    base = next(names_iter)
    ops = [Operation(kind="premise", content=base),
           Operation(kind="premise", content="-" + base),
           Operation(kind="prefer_premise", stronger="-" + base, weaker=base)]
    src = ("-" + base) if winner_feeds else base
    ridx[0] += 1
    rule = f"{prefix}_{ridx[0]}"
    out_lit = next(names_iter)
    ops.append(Operation(kind="defeasible", name=rule, antecedents=(src,), consequent=out_lit))
    return ops, out_lit, base, rule


def language_enrichment(names_iter, ridx, prefix="lx", contested_lit=None):
    from arggym.aspic.engine import Operation
    ops = []
    produced = []

    ax = next(names_iter)
    mid = next(names_iter)
    ops.append(Operation(kind="axiom", content=ax))
    ridx[0] += 1
    ops.append(Operation(kind="strict", name=f"{prefix}_{ridx[0]}", antecedents=(ax,),
                         consequent=mid))
    produced.append(mid)

    base = next(names_iter)
    out = next(names_iter)
    ops.append(Operation(kind="premise", content=base))
    ops.append(Operation(kind="premise", content="-" + base))
    ridx[0] += 1
    fired = f"{prefix}_{ridx[0]}"
    ops.append(Operation(kind="defeasible", name=fired, antecedents=("-" + base,), consequent=out))
    produced.append(out)

    r1, r2 = next(names_iter), next(names_iter)
    tgt = next(names_iter)
    ops.append(Operation(kind="premise", content=r1))
    ops.append(Operation(kind="premise", content=r2))
    ridx[0] += 1
    pro = f"{prefix}_{ridx[0]}"
    ops.append(Operation(kind="defeasible", name=pro, antecedents=(r1,), consequent=tgt))
    ridx[0] += 1
    con = f"{prefix}_{ridx[0]}"
    ops.append(Operation(kind="defeasible", name=con, antecedents=(r2,), consequent="-" + tgt))
    produced.append(tgt)

    ops.append(Operation(kind="prefer_premise", stronger="-" + base, weaker=base))
    ops.append(Operation(kind="prefer_rule", stronger=pro, weaker=con))
    return ops, produced


def _decompose(candidates, holds, probe_singletons=True):
    full = list(candidates)
    if not probe_singletons:
        return [list(range(len(full)))]
    n = len(full)
    useless = []
    for i in range(n):
        rest = [o for j, o in enumerate(full) if j != i]
        if rest and holds(rest):
            continue
        useless.append(i)
    return [list(range(n))]


def minimal_subset_exact(candidates, holds, max_calls=20000):
    full = list(candidates)
    calls = [0]

    def H(sub):
        calls[0] += 1
        return holds(list(sub))

    if not full or not H(full):
        return None, False, calls[0]

    keep = []
    for i, o in enumerate(full):
        rest = [x for j, x in enumerate(full) if j != i and x not in ()]
        if not H([x for j, x in enumerate(full) if j != i]):
            keep.append(o)
    forced = list(keep)
    lower = max(1, len(forced))
    if forced and H(forced):
        return forced, True, calls[0]

    optional = [o for o in full if o not in forced]
    best = list(full)
    proven = False
    from itertools import combinations
    for extra in range(0, len(optional) + 1):
        size = len(forced) + extra
        if size >= len(best):
            break
        exhausted = False
        for combo in combinations(optional, extra):
            if calls[0] > max_calls:
                exhausted = True
                break
            cand = forced + list(combo)
            if H(cand):
                best = cand
                proven = True
                break
        if proven or exhausted:
            break
    if not proven and calls[0] <= max_calls:
        proven = True
    return best, proven, calls[0]


def transpose_rule(op):
    from arggym.aspic.engine import Operation
    if op.kind != "strict":
        return []
    ants = list(op.antecedents or ())
    if not ants or not op.consequent:
        return []

    def neg(x):
        return x[1:] if x.startswith("-") else "-" + x

    out = []
    for i, a in enumerate(ants):
        rest = [x for j, x in enumerate(ants) if j != i]
        out.append(Operation(kind="strict", name=f"{op.name}_tp{i}",
                             antecedents=tuple([neg(op.consequent)] + rest),
                             consequent=neg(a)))
    return out


def close_under_transposition(ops):
    have = {(tuple(o.antecedents or ()), o.consequent)
            for o in ops if o.kind == "strict"}
    added = []
    for o in list(ops):
        if o.kind != "strict":
            continue
        for t in transpose_rule(o):
            key = (tuple(t.antecedents), t.consequent)
            if key not in have:
                have.add(key)
                added.append(t)
    return list(ops) + added, len(added)


def strict_conclusions_clash(ops):
    strict = [o for o in ops if o.kind == "strict" and o.consequent]
    out = []
    for i, a in enumerate(strict):
        for b in strict[i + 1:]:
            if a.consequent == ("-" + b.consequent) or b.consequent == ("-" + a.consequent):
                out.append((a.name, b.name))
    return out
