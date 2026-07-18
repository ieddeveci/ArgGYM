
from aspic_gym import (
    GYM_ORDERING, ATTACK_CHAIN_LEVEL, _rename_in_order, Operation, contrary, ASPICVerifier, JUSTIFIED, OVERRULED, load_kb, _n_rules, _pick_by_level, _gloss, _entry, _render_symbolic_theory, _render_content_theory,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl
import levels as difficulty

def _attack_symbolic(rng, level, req):
    depth = difficulty.knob('attack', 'depth', level)
    names = [f"{c}{i}" for c in "pqrstuvwxyz" for i in range(10)]; rng.shuffle(names); it = iter(names)
    fr = lambda: next(it)
    rl = [f"d{i}" for i in range(1, 40)]; rng.shuffle(rl); rit = iter(rl)
    sl = [f"s{i}" for i in range(1, 40)]; rng.shuffle(sl); sit = iter(sl)
    ops = []
    p0 = fr(); ops.append(Operation(kind="premise", content=p0))        
    ax0 = None
    if difficulty.gate('attack', 'axiom_distractor', level):
        ax0 = fr(); ops.append(Operation(kind="axiom", content=ax0)) 
    m = fr(); ant = (p0,) if ax0 is None else (p0, ax0)
    nm = next(rit); ops.append(Operation(kind="defeasible", name=nm, antecedents=ant, consequent=m))
    def_rules = [nm]; cur = m
    nrest = depth - 1
    strict_idx = rng.randrange(nrest) if (difficulty.gate('attack', 'strict_on_path', level) and nrest > 0) else -1
    if req in ("rebut", "outprefer") and strict_idx == nrest - 1:   
        strict_idx = rng.randrange(nrest - 1) if nrest > 1 else -1
    use_neg = level >= 4
    for i in range(nrest):
        nxt = fr()
        neg = use_neg and rng.random() < 0.4
        cons = ("-" + nxt) if neg else nxt
        if i == strict_idx:
            snm = next(sit)
            ops.append(Operation(kind="strict", name=snm, antecedents=(cur,), consequent=cons))
        else:
            dnm = next(rit)
            ops.append(Operation(kind="defeasible", name=dnm, antecedents=(cur,), consequent=cons))
            def_rules.append(dnm)
        cur = cons
    C = cur
    if difficulty.gate('attack', 'side_branch', level):                 
        q0 = fr(); z0 = fr(); qnm = next(rit)
        ops.append(Operation(kind="premise", content=q0))
        ops.append(Operation(kind="defeasible", name=qnm, antecedents=(q0,), consequent=z0))
    chain = level >= ATTACK_CHAIN_LEVEL and req in ("rebut", "undercut")
    b = fr() if chain else None
    rng.shuffle(ops)
    ops, name_map = _rename_in_order(ops)
    def_rules = [name_map[r] for r in def_rules]
    if req == "undermine":
        claim = contrary(p0)
    elif req == "undercut":
        claim = contrary(def_rules[-1])
    elif req == "outprefer":
        claim = contrary(C)
        if rng.random() < 0.5:
            witness = f"[premise: {contrary(p0)}][prefer_premise: {contrary(p0)} > {p0}]"
        else:
            top = next(o.name for o in ops if o.kind == "defeasible" and o.consequent == C)
            dn = sum(1 for o in ops if o.kind == "defeasible")
            b2 = fr()
            witness = (f"[premise: {b2}][defeasible: {b2} => {claim}]"
                       f"[prefer_rule: d{dn + 1} > {top}]")
        return ops, C, GYM_ORDERING, witness
    else:
        claim = contrary(C)
    if chain:                                  
        witness = f"[premise: {b}][defeasible: {b} => {claim}]"
    else:
        witness = f"[premise: {claim}]"          
    return ops, C, GYM_ORDERING, witness


def _attack_content(rng, level, req):
    kb = load_kb()
    recs = [r for r in kb if r.get("support")]
    rng.shuffle(recs)
    allow_ax = difficulty.gate('attack', 'allow_ax', level); allow_strict = difficulty.gate('attack', 'allow_strict', level)
    cands = []
    for rec in recs:
        if len(cands) >= 50:
            break
        atoms = rec["atoms"]
        args = rec["support"][:]; rng.shuffle(args)
        for arg in args:
            ops = []; ord_prems = []
            for lit in arg["premises"]:
                ops.append(Operation(kind="premise", content=lit)); ord_prems.append(lit)
            for lit in arg.get("axioms", []):
                k = "axiom" if (allow_ax and atoms.get(lit.lstrip("-"), {}).get("axiomatic")) else "premise"
                ops.append(Operation(kind=k, content=lit))
                if k == "premise":
                    ord_prems.append(lit)
            for r in arg["rules"]:
                k = "strict" if (allow_strict and r.get("strict")) else "defeasible"
                ops.append(Operation(kind=k, name=r["name"], antecedents=tuple(r["antecedents"]),
                                     consequent=r["consequent"]))
            C = arg["conclusion"]
            if req == "undermine" and not ord_prems:
                continue
            v = ASPICVerifier.from_operations(ops, ordering="last_link_elitist")
            if v.status(C) != JUSTIFIED:
                continue
            if req == "undercut":
                drs = [o for o in ops if o.kind == "defeasible"]
                rng.shuffle(drs)
                rules_disp = [o for o in ops if o.kind in ("defeasible", "strict")]
                hit = None
                for dr in drs:
                    cl = contrary(dr.name)
                    vv = ASPICVerifier.from_operations(ops, ordering="last_link_elitist")
                    try:
                        vv.fw.apply(Operation(kind="premise", content=cl))
                    except Exception:
                        continue
                    if vv.status(C) != JUSTIFIED:
                        k = next(i for i, o in enumerate(rules_disp, 1) if o.name == dr.name)
                        if level >= ATTACK_CHAIN_LEVEL and ord_prems:   
                            hit = f"[defeasible: {_gloss(atoms, ord_prems[0])} => -Rule {k}]"
                        else:
                            hit = f"[premise: -Rule {k}]"
                        break
                if not hit:
                    continue
                cands.append((atoms, ops, C, hit)); continue
            if req == "outprefer":
                rules_disp = [o for o in ops if o.kind in ("defeasible", "strict")]
                tops = [o for o in ops if o.kind == "defeasible" and o.consequent == C]
                claim = contrary(C)
                cbase = claim.lstrip("-")
                claim_txt = ("-" + atoms[cbase]["pos"]) if claim.startswith("-") else atoms[cbase]["pos"]
                made = None
                if rng.random() < 0.5 and ord_prems:
                    p = ord_prems[0]
                    v2 = ASPICVerifier.from_operations(ops, ordering="last_link_elitist")
                    try:
                        v2.fw.apply(Operation(kind="premise", content=contrary(p)))
                        v2.fw.apply(Operation(kind="prefer_premise", stronger=contrary(p), weaker=p))
                        if v2.status(C) == OVERRULED:
                            made = (f"[premise: {_gloss(atoms, contrary(p))}]\n"
                                    f"[prefer_premise: {_gloss(atoms, contrary(p))} > {_gloss(atoms, p)}]")
                    except Exception:
                        pass
                if made is None and tops and ord_prems:
                    top = tops[0]
                    k = next(i for i, o in enumerate(rules_disp, 1) if o.name == top.name)
                    v2 = ASPICVerifier.from_operations(ops, ordering="last_link_elitist")
                    try:
                        v2.fw.apply(Operation(kind="defeasible", name="_x",
                                              antecedents=(ord_prems[0],), consequent=claim))
                        v2.fw.apply(Operation(kind="prefer_rule", stronger="_x", weaker=top.name))
                        if v2.status(C) == OVERRULED:
                            made = (f"[defeasible: {_gloss(atoms, ord_prems[0])} => {claim_txt}]\n"
                                    f"[prefer_rule: Rule {len(rules_disp) + 1} > Rule {k}]")
                    except Exception:
                        pass
                if not made:
                    continue
                cands.append((atoms, ops, C, made)); continue
            tgt = ord_prems[0] if req == "undermine" else C
            claim = contrary(tgt)
            v2 = ASPICVerifier.from_operations(ops, ordering="last_link_elitist")
            try:
                v2.fw.apply(Operation(kind="premise", content=claim))
            except Exception:
                continue
            if v2.status(C) == JUSTIFIED:               
                continue
            cbase = claim.lstrip("-")
            claim_txt = ("-" + atoms[cbase]["pos"]) if claim.startswith("-") else atoms[cbase]["pos"]
            if req == "rebut" and level >= ATTACK_CHAIN_LEVEL and ord_prems:   
                wit = f"[defeasible: {_gloss(atoms, ord_prems[0])} => {claim_txt}]"
            else:
                wit = f"[premise: {claim_txt}]"
            cands.append((atoms, ops, C, wit))
    return _pick_by_level(cands, level, rng, key=lambda c: _n_rules(c[1]), tag=None) if cands else None


def frame_attack(rng, with_content=False, level=2):
    req = rng.choice(["undermine", "rebut", "undercut"]
                     + (["outprefer"] if level >= 3 else []))
    if with_content:
        built = _attack_content(rng, level, req)
        if not built:
            return None
        atoms, ops, C, witness = built
        ordering = GYM_ORDERING; gloss = lambda l: _gloss(atoms, l)
        theory_text = _render_content_theory(atoms, ops); notation = _CONTENT_NOTATION; mode = "content"
        how = ("Write your attacking argument as bracketed directives, one per line, in this syntax: "
               "[premise: a statement], [defeasible: A AND B => C], [strict: A AND B -> C], using the "
               "statements as they read in the theory. To NEGATE a statement, prefix it with the "
               "negation operator '-', for example [premise: -the sky is blue] (a natural-language "
               "negation is also accepted). To switch off a rule (undercut), negate its label with "
               "'-', e.g. [premise: -Rule 2] (this means Rule 2 no longer applies).")
        meta_extra = {"atoms": atoms}
    else:
        ops, C, ordering, witness = _attack_symbolic(rng, level, req)
        gloss = lambda l: l
        theory_text = _render_symbolic_theory(ops); notation = _sym_notation(level); mode = "symbolic"
        how = ("Write your attacking argument as bracketed directives, one per line, in this syntax: "
               "[premise: x], [defeasible: a AND b => c], [strict: a AND b -> c]. The negation of a "
               "statement x is written -x; to rebut a conclusion or undermine a premise, derive or "
               "assert its negation. To undercut a rule, assert the negation of its label, e.g. "
               "[premise: -d2].")
        meta_extra = {}
    kindword = {"undermine": "UNDERMINE - attack a premise the argument relies on",
                "rebut": "REBUT - attack one of its defeasible conclusions",
                "undercut": "UNDERCUT - switch off one of its defeasible rules",
                "outprefer": ("OUT-PREFER - attack it (rebut a conclusion or undermine a premise) "
                              "AND add a preference directive so your attacker prevails")}[req]
    goal_txt = ("becomes OVERRULED (decisively defeated; a mere deadlock where both sides are "
                "undecided is NOT enough)" if req == "outprefer" else "is no longer justified")
    pref_hint = (" Preference directives: [prefer_rule: X > Y] (X, Y rule labels) and "
                 "[prefer_premise: x > y]." if req == "outprefer" else "")
    chain = level >= ATTACK_CHAIN_LEVEL and req in ("rebut", "undercut")
    chain_hint = (" Do not simply assert the contrary as a premise; derive your attacking conclusion "
                  "through at least one rule (a chain)." if chain else "")
    task = ("You are analysing a defeasible argumentation theory. " + notation + "\n\nTheory:\n"
            + theory_text + f"\n\nThe argument whose conclusion is \"{gloss(C)}\" is currently "
            f"JUSTIFIED.\n\nConstruct an argument that mounts a successful {kindword}, so that "
            f"\"{gloss(C)}\" {goal_txt}." + chain_hint + pref_hint + " " + how)
    spec = ("Decide your strategy (which point to attack and how) in the reasoning block. The answer "
            "block must contain the directives of your attacking argument, one per line, with no "
            "prose. Keep the combined theory consistent.")
    prompt = _intro(level) + "\n\n" + _ordering_decl(ordering) + task + "\n\n" + _format_block(spec)
    ref = "[answer]\n" + witness + "\n[/answer]"
    return _entry("attack", prompt, ref, ops, ordering, target=C, req=req, mode=mode,
                  level=level, **meta_extra)
