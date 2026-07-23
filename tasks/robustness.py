from dataclasses import replace

from aspic_gym import (
    GYM_ORDERING, ASPICVerifier, Operation, JUSTIFIED, _entry, _gloss, _kb_gloss_pool, _rename_in_order, _render_symbolic_theory, _render_content_theory, ops_to_dicts, sanitize_statement,
)
from prompting import _format_block, _sym_notation, _CONTENT_NOTATION
import levels as difficulty

def _variant_at(level):
    return difficulty.recipe('robustness_variant', level)


def _status_without(ops, ordering, target, removed):
    v = ASPICVerifier.from_operations(
        [o for o in ops if not (o.kind == "premise" and o.content in removed)],
        ordering=ordering)
    return str(v.status(target))


def _achieves(ops, ordering, target, removed, direction):
    s = _status_without(ops, ordering, target, removed)
    return (s == "JUSTIFIED") if direction == "reinstate" else (s != "JUSTIFIED")


def _critical_singles(ops, ordering, target, direction):
    prem = [o.content for o in ops if o.kind == "premise"]
    return [p for p in prem if _achieves(ops, ordering, target, {p}, direction)]


def _build_ops(rng, level, direction="defeat"):
    i, ri = [0], [0]
    def fr():
        i[0] += 1
        return f"{'pqrstuvwxy'[rng.randrange(10)]}{i[0]}"
    P = lambda c: Operation(kind="premise", content=c)
    def D(a, c):
        ri[0] += 1
        return Operation(kind="defeasible", name=f"t{ri[0]}", antecedents=tuple(a),
                         consequent=c)
    a, b, k, c0 = fr(), fr(), fr(), fr()
    b1 = fr()
    ops = [P(a), P(b), P(k), D([a], b1), D([b], b1)]

    if level == 1:                                 
        ops.append(D([b1, k], c0))
        ops.append(P(fr()))
        return ops, c0

    if level == 2:                                  
        m2 = fr()                                    
        ops.append(D([b1, k], m2))
        ops.append(D([m2], c0))
        side = fr()
        ops.append(D([b1], "-" + side))
        ops += [P(fr()), P(fr())]
        return ops, c0

    if level == 3:                                  
        m2, k2 = fr(), fr()
        ops.append(D([b1, k], m2))
        ops.append(P(k2))
        ops.append(D([m2, k2], c0))
        ops += [P(fr()), P(fr())]
        return ops, c0

    if level == 4:                                    
        ops.append(D([b1, k], c0))
        vsrc = fr()
        ops.append(P(vsrc))
        ops.append(D([vsrc], "-" + c0))            
        ops += [P(fr()), P(fr())]
        return ops, c0

    m2 = fr()                                        
    ops.append(D([b1], m2))                         
    ops.append(D([m2], c0))                           
    vsrc = fr()
    ops.append(P(vsrc))
    d_att = D([vsrc], "-" + c0)
    ops.append(d_att)
    sup_rule = next(o for o in ops if o.kind == "defeasible" and o.consequent == c0)
    # Which rule wins decides which question the item can pose. Preferring the
    # support leaves the target justified, so the task is "retract something to
    # defeat it". Preferring the attacker leaves it overruled, so retracting the
    # attacker's premise restores it -- the reinstatement case, which is
    # otherwise unreachable above level 4 and so went untested entirely.
    strong, weak = ((d_att.name, sup_rule.name) if direction == "reinstate"
                    else (sup_rule.name, d_att.name))
    ops.append(Operation(kind="prefer_rule", stronger=strong, weaker=weak))
    ops += [P(fr())]
    return ops, c0


def _guards_ok(ops, ordering, target, variant, direction):
    prem = [o.content for o in ops if o.kind == "premise"]
    singles = _critical_singles(ops, ordering, target, direction)
    if variant == "single":
        return 1 <= len(singles) < len(prem)
    if variant == "set":
        return 2 <= len(singles) < len(prem)
    if singles:
        return False
    pairs = [(x, y) for i, x in enumerate(prem) for y in prem[i + 1:]
             if _achieves(ops, ordering, target, {x, y}, direction)]
    return bool(pairs)


_Q = {
    ("single", "defeat"):
        "{t} is currently JUSTIFIED. Name ONE ordinary premise whose retraction makes {t} "
        "no longer justified.",
    ("set", "defeat"):
        "{t} is currently JUSTIFIED. List EVERY ordinary premise whose INDIVIDUAL retraction "
        "makes {t} no longer justified.",
    ("single", "reinstate"):
        "{t} is currently NOT justified. Name ONE ordinary premise whose retraction MAKES {t} "
        "justified.",
    ("pair", "defeat"):
        "{t} is currently JUSTIFIED, and no single retraction changes that. Name TWO ordinary "
        "premises whose JOINT retraction makes {t} no longer justified - and neither may "
        "suffice alone.",
}
_SPEC = {
    "single": "In the answer, name exactly one premise statement and nothing else.",
    "set": "In the answer, list the premise statements separated by commas (or one per line), "
           "and nothing else.",
    "pair": "In the answer, name exactly two premise statements, separated by a comma, and "
            "nothing else.",
}


def frame_robustness(rng, with_content=False, level=1):
    variant, direction = _variant_at(level)
    for _ in range(40):
        ops, target = _build_ops(rng, level, direction)
        zi = 100
        for _pad in range(min(max(0, level - 2), 12)):
            zi += 1
            ops.append(Operation(kind="premise", content=f"z{zi}"))
        rng.shuffle(ops)
        ops, name_map = _rename_in_order(ops)
        ops = [replace(o, stronger=name_map.get(o.stronger, o.stronger),
                       weaker=name_map.get(o.weaker, o.weaker))
               if o.kind in ("prefer_rule", "prefer_premise") else o for o in ops]
        ops = [o for o in ops if o.kind not in ("prefer_rule", "prefer_premise")] + \
              [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
        ordering = GYM_ORDERING
        v = ASPICVerifier.from_operations(ops, ordering=ordering)
        base_just = str(v.status(target)) == "JUSTIFIED"
        if direction == "defeat" and not base_just:
            continue
        if direction == "reinstate" and base_just:
            continue
        if not _guards_ok(ops, ordering, target, variant, direction):
            continue

        prem = [o.content for o in ops if o.kind == "premise"]
        singles = _critical_singles(ops, ordering, target, direction)
        if variant == "single":
            gold = [singles[rng.randrange(len(singles))]]
        elif variant == "set":
            gold = sorted(singles)
        else:
            pairs = [(x, y) for i, x in enumerate(prem) for y in prem[i + 1:]
                     if _achieves(ops, ordering, target, {x, y}, "defeat")]
            gold = sorted(pairs[rng.randrange(len(pairs))])

        if with_content:
            pool = _kb_gloss_pool()
            if not pool:
                return None
            syms = []
            for o in ops:
                for l in ([o.content] if o.content else []) + list(o.antecedents) + \
                         ([o.consequent] if o.consequent else []):
                    b = l.lstrip("-")
                    if not (b[:1] in "ds" and b[1:].isdigit()) and b not in syms:
                        syms.append(b)
            if len(pool) < len(syms):
                return None
            picks = rng.sample(pool, len(syms))
            atoms = {s: {"pos": picks[j]} for j, s in enumerate(syms)}
            gloss = lambda l: _gloss(atoms, l)
            theory_text = _render_content_theory(atoms, ops)
            notation = _CONTENT_NOTATION
        else:
            atoms = None
            gloss = lambda l: l
            theory_text = _render_symbolic_theory(ops)
            notation = _sym_notation(level)

        q = _Q[(variant, direction)].format(t=('"' + gloss(target) + '"') if with_content
                                            else target)
        prompt = ("You are working with a defeasible argumentation theory. " + notation
                  + "\n\nTheory:\n" + theory_text + "\n\n" + q + "\n\n"
                  + _format_block(_SPEC[variant]))
        ref_txt = ", ".join(gloss(g) for g in gold)
        ref = ref_txt
        meta = dict(target=target, variant=variant, direction=direction,
                    critical_set=sorted(singles), gold_names=gold,
                    mode="content" if with_content else "symbolic")
        if with_content:
            meta["atoms"] = atoms
        return _entry("robustness", prompt, ref, ops, ordering, **meta)
    return None
