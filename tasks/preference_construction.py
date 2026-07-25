from dataclasses import replace

from aspic_gym import (
    GYM_ORDERING, _rename_in_order, Operation, contrary, ASPICVerifier, JUSTIFIED, load_kb, _ops_from_args, _gloss, _entry, _render_symbolic_theory, _render_content_theory,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl
import levels as difficulty


def _bands_at(level):
    k = lambda n: difficulty.knob('preference_construction', n, level)
    return (k('depth_min'), k('depth_max'), k('con_min'), k('con_max'),
            k('decoy_min'), k('decoy_max'), k('f_axiom'), k('f_strict'), k('f_undercut'))


def _chain(fr, frr, depth):
    p0 = fr()
    ops = [Operation(kind="premise", content=p0)]
    cur, nodes, rules = p0, [], []
    for _ in range(depth):
        nxt = fr(); nm = frr()
        ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
        nodes.append(nxt); rules.append(nm); cur = nxt
    return ops, nodes, rules


def _tpl_direct(rng, depth, fr, frr):
    ops, nodes, rules = _chain(fr, frr, depth)
    T = nodes[-1]
    pi = fr(); ai = frr()
    ops += [Operation(kind="premise", content=pi),
            Operation(kind="defeasible", name=ai, antecedents=(pi,), consequent=contrary(T))]
    return ops, T, [("rule", rules[-1], ai)]


def _tpl_upstream(rng, depth, fr, frr):
    depth = max(2, depth)
    ops, nodes, rules = _chain(fr, frr, depth)
    T = nodes[-1]
    k = rng.randrange(depth - 1)                
    pi = fr(); ai = frr()
    ops += [Operation(kind="premise", content=pi),
            Operation(kind="defeasible", name=ai, antecedents=(pi,), consequent=contrary(nodes[k]))]
    return ops, T, [("rule", rules[k], ai)]


def _tpl_intermediate_target(rng, depth, fr, frr):
    depth = max(2, depth)
    ops, nodes, rules = _chain(fr, frr, depth)
    k = rng.randrange(depth - 1)              
    T = nodes[k]
    pi = fr(); ai = frr()
    ops += [Operation(kind="premise", content=pi),
            Operation(kind="defeasible", name=ai, antecedents=(pi,), consequent=contrary(T))]
    return ops, T, [("rule", rules[k], ai)]


def _tpl_two_supports(rng, depth, fr, frr):
    p1, p2, T = fr(), fr(), fr()
    r1, r2 = frr(), frr()
    ops = [Operation(kind="premise", content=p1), Operation(kind="premise", content=p2),
           Operation(kind="defeasible", name=r1, antecedents=(p1,), consequent=T),
           Operation(kind="defeasible", name=r2, antecedents=(p2,), consequent=T)]
    pi = fr(); ai = frr()
    ops += [Operation(kind="premise", content=pi),
            Operation(kind="defeasible", name=ai, antecedents=(pi,), consequent=contrary(T))]
    return ops, T, [("rule", r1, ai), ("rule", r2, ai)]


def _tpl_reinstatement(rng, depth, fr, frr):
    p0, T, a, pa, pb = fr(), fr(), fr(), fr(), fr()
    r1, rU, rA, rB = frr(), frr(), frr(), frr()
    ops = [Operation(kind="premise", content=p0),
           Operation(kind="defeasible", name=r1, antecedents=(p0,), consequent=T),
           Operation(kind="premise", content=pa),
           Operation(kind="defeasible", name=rA, antecedents=(pa,), consequent=a),
           Operation(kind="premise", content=pb),
           Operation(kind="defeasible", name=rB, antecedents=(pb,), consequent=contrary(a)),
           Operation(kind="defeasible", name=rU, antecedents=(a,), consequent=contrary(T))]
    return ops, T, [("rule", rB, rA)]


def _tpl_prefer_premise(rng, depth, fr, frr):
    x, T = fr(), fr()
    r1, rU = frr(), frr()
    ops = [Operation(kind="premise", content=x),
           Operation(kind="premise", content=contrary(x)),
           Operation(kind="defeasible", name=r1, antecedents=(x,), consequent=T),
           Operation(kind="defeasible", name=rU, antecedents=(contrary(x),), consequent=contrary(T))]
    return ops, T, [("premise", x, contrary(x))]


_PREF_BUILDERS = {
    "direct": _tpl_direct, "upstream": _tpl_upstream, "intermediate_target": _tpl_intermediate_target,
    "two_supports": _tpl_two_supports, "reinstatement": _tpl_reinstatement,
    "prefer_premise": _tpl_prefer_premise,
}
def _templates_at(level):
    return difficulty.recipe('preference_templates', level)


def _augment_premise_prefs(ops, wpairs):
    cons_map = {}
    for o in ops:
        if o.kind in ("defeasible", "strict"):
            cons_map.setdefault(o.consequent, o)
    rule_by_name = {o.name: o for o in ops if o.kind in ("defeasible", "strict") and o.name}
    ordinary = {o.content for o in ops if o.kind == "premise"}

    def prems_of_rule(rname):
        r = rule_by_name.get(rname)
        if not r:
            return set()
        prems, stack, seen = set(), list(r.antecedents or ()), set()
        while stack:
            lit = stack.pop()
            if lit in seen:
                continue
            seen.add(lit)
            if lit in ordinary:
                prems.add(lit)
            sub = cons_map.get(lit)
            if sub is not None:
                stack.extend(sub.antecedents or ())
        return prems

    out, seen_pairs = [], set()
    for k, s, w in wpairs:
        if k != "rule":
            continue
        for sp in sorted(prems_of_rule(s)):
            for wp in sorted(prems_of_rule(w)):
                if sp != wp and (sp, wp) not in seen_pairs and (wp, sp) not in seen_pairs:
                    seen_pairs.add((sp, wp))
                    out.append((sp, wp))
    return out


def _pref_symbolic(rng, level, ordering=None):
    ordering = ordering or "last_link_elitist"
    dmn, dmx, cmn, cmx, kmn, kmx, f_ax, f_st, f_uc = _bands_at(level)
    templates = _templates_at(level)
    for _attempt in range(200):
        names = [f"{c}{i}" for c in "pqrstuvwxyz" for i in range(10)]
        rng.shuffle(names); nit = iter(names)
        rlabels = [f"d{i}" for i in range(1, 300)]
        rng.shuffle(rlabels); rit = iter(rlabels)
        fr = lambda: next(nit)
        frr = lambda: next(rit)
        tpl = rng.choice(templates)
        depth = rng.randint(dmn, dmx)
        ops, T, wpairs = _PREF_BUILDERS[tpl](rng, depth, fr, frr)
        t_rules = [o.name for o in ops if o.kind in ("defeasible", "strict") and o.consequent == T]
        ncon = rng.randint(cmn, cmx)
        _guard = 0
        while len(wpairs) < ncon and t_rules and _guard < 6:
            pi = fr(); ai = frr()
            ops += [Operation(kind="premise", content=pi),
                    Operation(kind="defeasible", name=ai, antecedents=(pi,), consequent=contrary(T))]
            for tr in t_rules:
                wpairs.append(("rule", tr, ai))
            _guard += 1
        if f_ax:                                 
            af, bf = fr(), fr()
            ops += [Operation(kind="axiom", content=af),
                    Operation(kind="defeasible", name=frr(), antecedents=(af,), consequent=bf)]
        if f_st:                                 
            sp, se = fr(), fr()
            ops += [Operation(kind="premise", content=sp),
                    Operation(kind="strict", name=frr(), antecedents=(sp,), consequent=se)]
        for _ in range(rng.randint(kmn, kmx)):   
            d1, d2, z = fr(), fr(), fr()
            ops += [Operation(kind="premise", content=d1), Operation(kind="premise", content=d2),
                    Operation(kind="defeasible", name=frr(), antecedents=(d1,), consequent=z),
                    Operation(kind="defeasible", name=frr(), antecedents=(d2,), consequent=contrary(z))]
        if f_uc:                                  
            up, ut, ucp = fr(), fr(), fr()
            ur = frr()
            ops += [Operation(kind="premise", content=up),
                    Operation(kind="defeasible", name=ur, antecedents=(up,), consequent=ut),
                    Operation(kind="premise", content=ucp),
                    Operation(kind="defeasible", name=frr(), antecedents=(ucp,), consequent="-" + ur)]
        rng.shuffle(ops)
        ops, name_map = _rename_in_order(ops)
        remap = lambda l: ("-" + name_map[l[1:]]) if (isinstance(l, str) and l.startswith("-")
                                                      and l[1:] in name_map) else l
        ops = [replace(o, content=remap(o.content)) if o.kind in ("premise", "axiom") else
               replace(o, antecedents=tuple(remap(a) for a in o.antecedents),
                       consequent=remap(o.consequent)) if o.kind in ("defeasible", "strict")
               else o for o in ops]
        wpairs = [(k, name_map.get(s, s), name_map.get(w, w)) for k, s, w in wpairs]
        prem_pairs = _augment_premise_prefs(ops, wpairs)
        witness = "\n".join(
            [f"[prefer_{'rule' if k == 'rule' else 'premise'}: {s} > {w}]" for k, s, w in wpairs]
            + [f"[prefer_premise: {s} > {w}]" for s, w in prem_pairs])
        try:
            v0 = ASPICVerifier.from_operations(ops, ordering=ordering)
            if v0.status(T) == JUSTIFIED:
                continue
            v1 = ASPICVerifier.from_operations(ops, ordering=ordering)
            for k, s, w in wpairs:
                v1.fw.apply(Operation(kind=("prefer_rule" if k == "rule" else "prefer_premise"),
                                      stronger=s, weaker=w))
            for s, w in prem_pairs:
                v1.fw.apply(Operation(kind="prefer_premise", stronger=s, weaker=w))
            if v1.status(T) == JUSTIFIED and v1.is_consistent():
                return ops, T, ordering, witness, tpl
        except Exception:
            continue
    p0, t0, p1 = "p0", "t0", "p1"           
    ops = [Operation(kind="premise", content=p0),
           Operation(kind="defeasible", name="d1", antecedents=(p0,), consequent=t0),
           Operation(kind="premise", content=p1),
           Operation(kind="defeasible", name="d2", antecedents=(p1,), consequent="-" + t0)]
    fb = "[prefer_rule: d1 > d2]\n[prefer_premise: p0 > p1]"
    return ops, t0, ordering, fb, "direct"

def _pref_content(rng, level):
    kb = load_kb()
    if not kb:
        return None
    dmn, dmx, cmn, cmx, kmn, kmx, f_ax, f_st, f_uc = _bands_at(level)
    contra = lambda x: x[1:] if x.startswith("-") else "-" + x
    avail = [t for t in _templates_at(level)
             if t in ("direct", "intermediate_target", "upstream", "two_supports",
                      "prefer_premise")] or ["direct"]
    recs = [r for r in kb if r.get("support")]
    rng.shuffle(recs)
    for rec in recs[:80]:
        atoms = rec["atoms"]; sup = rec["support"]
        tr = difficulty.knob('kb_theory', 'rules', level)
        pool = sorted(sup, key=lambda a: len(a["rules"]))
        picked, run = [], 0
        for a in pool:
            if picked and run >= tr:
                break
            picked.append(a); run += len(a["rules"])
        args = tuple(picked)
        ops = _ops_from_args(args, atoms)
        if not ops:
            continue
        gated = []                             
        for o in ops:
            if o.kind == "axiom" and not f_ax:
                o = Operation(kind="premise", content=o.content)
            elif o.kind == "strict" and not f_st:
                o = Operation(kind="defeasible", name=o.name, antecedents=o.antecedents,
                              consequent=o.consequent)
            gated.append(o)
        ops = gated
        nr = sum(1 for o in ops if o.kind in ("defeasible", "strict"))
        if nr < 1 or nr > max(10, difficulty.knob('kb_theory', 'rules', level) + 4):
            continue
        try:
            sm = ASPICVerifier.from_operations(ops, ordering="last_link_elitist").status_map()
        except Exception:
            continue
        by_cons = {}
        for o in ops:
            if o.kind in ("defeasible", "strict"):
                by_cons.setdefault(o.consequent, []).append(o)
        def _anc(x, seen):
            for r in by_cons.get(x, []):
                for a in r.antecedents:
                    if a not in seen:
                        seen.add(a); _anc(a, seen)
            return seen
        uniq = {c: rs[0].name for c, rs in by_cons.items()
                if sm.get(c) == JUSTIFIED and len(rs) == 1 and rs[0].kind == "defeasible"}
        doub = {c: [r.name for r in rs] for c, rs in by_cons.items()
                if sm.get(c) == JUSTIFIED and len(rs) == 2
                and all(r.kind == "defeasible" for r in rs)}
        leaves = [o.content for o in ops if o.kind in ("premise", "axiom")]
        if not leaves:
            continue
        tpl = rng.choice(avail)
        extra, witness_pairs, target = [], [], None
        prem_pairs = []                              
        drives = leaves[:]; rng.shuffle(drives)

        if tpl == "prefer_premise":
            ordinary = {o.content for o in ops if o.kind == "premise"}
            opts = []
            for t, _rn in uniq.items():
                # sorted: _anc returns a set, and rng.choice below indexes into
                # this list, so raw set order would make the generated item depend
                # on PYTHONHASHSEED -- which differs per taskset-build worker.
                deps = sorted(l for l in _anc(t, set()) if l in ordinary)
                if deps:
                    opts.append((t, deps))
            if opts:
                target, deps = rng.choice(opts)
                p = rng.choice(deps)
                extra = [Operation(kind="premise", content=contra(p))]
                prem_pairs = [(p, contra(p))]

        if tpl == "two_supports" and doub:
            target = rng.choice(list(doub)); rnames = doub[target]
            extra = [Operation(kind="defeasible", name="_cmp0", antecedents=(drives[0],),
                               consequent=contra(target))]
            witness_pairs = [(rn, "_cmp0") for rn in rnames]
        elif tpl == "upstream":
            opts = [(t, sorted(_anc(t, set()) & set(uniq))) for t in uniq]  
            opts = [(t, mids) for t, mids in opts if mids]
            if opts:
                target, mids = rng.choice(opts); mid = rng.choice(mids)
                extra = [Operation(kind="defeasible", name="_cmp0", antecedents=(drives[0],),
                                   consequent=contra(mid))]
                witness_pairs = [(uniq[mid], "_cmp0")]
        elif tpl == "intermediate_target":
            inner = [t for t in uniq if any(t in _anc(o, set()) for o in uniq if o != t)]
            if inner:
                target = rng.choice(inner); rname = uniq[target]
                n = min(rng.randint(cmn, cmx), len(drives))
                for i in range(n):
                    extra.append(Operation(kind="defeasible", name=f"_cmp{i}",
                                           antecedents=(drives[i],), consequent=contra(target)))
                    witness_pairs.append((rname, f"_cmp{i}"))
        if target is None:                           
            tpl = "direct"
            if not uniq:
                continue
            target = rng.choice(list(uniq)); rname = uniq[target]
            n = min(rng.randint(cmn, cmx), len(drives))
            for i in range(n):
                extra.append(Operation(kind="defeasible", name=f"_cmp{i}",
                                       antecedents=(drives[i],), consequent=contra(target)))
                witness_pairs.append((rname, f"_cmp{i}"))

        theory = ops + extra
        try:
            v0 = ASPICVerifier.from_operations(theory, ordering="last_link_elitist")
            if v0.status(target) == JUSTIFIED:
                continue
            v1 = ASPICVerifier.from_operations(theory, ordering="last_link_elitist")
            for s, w in witness_pairs:
                v1.fw.apply(Operation(kind="prefer_rule", stronger=s, weaker=w))
            for s, w in prem_pairs:
                v1.fw.apply(Operation(kind="prefer_premise", stronger=s, weaker=w))
            if not (v1.status(target) == JUSTIFIED and v1.is_consistent()):
                continue
        except Exception:
            continue
        rules_disp = [o for o in theory if o.kind in ("defeasible", "strict")]
        idx = {o.name: i + 1 for i, o in enumerate(rules_disp)}
        wlines = [f"[prefer_rule: Rule {idx[s]} > Rule {idx[w]}]" for s, w in witness_pairs]
        wlines += [f"[prefer_premise: {_gloss(atoms, s)} > {_gloss(atoms, w)}]"
                   for s, w in prem_pairs]
        witness = "\n".join(wlines)
        return atoms, theory, target, witness, tpl
    return None

def frame_preference_construction(rng, with_content=False, level=2, ordering=None):
    if with_content:
        built = _pref_content(rng, level)
        if not built:
            return None
        atoms, ops, T, witness, template = built
        ordering = GYM_ORDERING; gloss = lambda l: _gloss(atoms, l)
        theory_text = _render_content_theory(atoms, ops); notation = _CONTENT_NOTATION; mode = "content"
        how = ("State your preference(s) as directives, one per line, referring to rules by their "
               "number: [prefer_rule: Rule i > Rule j] means Rule i is preferred over Rule j.")
        meta_extra = {"atoms": atoms}
    else:
        ops, T, ordering, witness, template = _pref_symbolic(rng, level, ordering)
        gloss = lambda l: l
        theory_text = _render_symbolic_theory(ops); notation = _sym_notation(level); mode = "symbolic"
        how = ("State your preference(s) as directives, one per line: [prefer_rule: d1 > d2] means "
               "rule d1 is preferred over rule d2. You may also write [prefer_premise: x > y].")
    weak = "weakest_link" in (ordering or "")
    order_decl = (
        "ORDERING: this problem uses WEAKEST-LINK ordering -- an argument is only as strong as its "
        "weakest defeasible rule, so to make an argument win you must rank BOTH its rules AND its "
        "premises above the competing argument's.\n\n" if weak else
        "ORDERING: this problem uses LAST-LINK ordering -- an argument's strength is decided by its "
        "final defeasible rule.\n\n")
    task = ("You are analysing a defeasible argumentation theory. " + notation + "\n\nTheory:\n"
            + theory_text + f"\n\nThe statement  {gloss(T)}  is currently NOT justified: its "
            "supporting argument ties with one or more competing arguments and grounded semantics "
            "leaves it undecided. Add the preference(s) needed to break the tie(s) so that "
            + f"{gloss(T)} becomes justified (other, unrelated conflicts in the theory may be left "
            "as they are). " + how)
    spec = ("Work out which rule must win in the reasoning block. The answer block must contain only "
            "your preference directives, one per line.")
    prompt = _intro(level) + "\n\n" + order_decl + task + "\n\n" + _format_block(spec)
    ref = witness
    meta = {"target": T, "mode": mode, "n_prefs": witness.count("prefer_"), "template": template}
    if with_content:
        meta["atoms"] = atoms
    return _entry("preference_construction", prompt, ref, ops, ordering, **meta)
