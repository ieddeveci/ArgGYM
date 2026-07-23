
from aspic_gym import (
    GYM_ORDERING, Operation, contrary, ASPICVerifier, JUSTIFIED, load_kb, task_elements, _ops_from_args, _n_rules, _pick_by_level, _fresh_syms, _rename_in_order, _gloss, _entry, _render_symbolic_theory, _render_content_theory, register_cache_resetter,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl


def _build_ksupport(rng, k, level=2):
    syms = _fresh_syms(rng, k + 1)
    T = syms[k]
    if level >= 4 and rng.random() < 0.4:
        T = "-" + T
    ops = []
    for i in range(k):
        p = syms[i]
        ops.append(Operation(kind="premise", content=p))
        ops.append(Operation(kind="defeasible", name=f"d{i+1}", antecedents=(p,), consequent=T))
    rng.shuffle(ops)
    ops, _ = _rename_in_order(ops)                    
    supports = [o.name for o in ops if o.kind == "defeasible" and o.consequent == T]
    prem_of = {}                              
    for o in ops:
        if o.kind == "defeasible" and o.consequent == T and o.antecedents:
            prem_of[o.name] = o.antecedents[0]
    parts = []
    cT = contrary(T)
    use_pref = level >= 3
    parts.append(f"[premise: ww][defeasible cc: ww => {cT}]")
    outpref_rules = []
    premise_done = False
    for i, d in enumerate(supports):
        mode = rng.random()
        if use_pref and not premise_done and prem_of.get(d) and mode < 0.34:
            p = prem_of[d]
            parts.append(f"[premise: {contrary(p)}][prefer_premise: {contrary(p)} > {p}]")
            parts.append(f"[defeasible cp{i}: {contrary(p)} => {cT}]")
            premise_done = True
        elif use_pref and mode < 0.67:
            outpref_rules.append(d)
        else:
            parts.append(f"[premise: u{i}][defeasible: u{i} => -{d}]")
    for d in outpref_rules:
        parts.append(f"[prefer_rule: cc > {d}]")
    witness = "".join(parts)
    return ops, T, supports, witness

_COUNTER_CONTENT_POOL = None


def _reset_counter_content_pool():
    global _COUNTER_CONTENT_POOL
    _COUNTER_CONTENT_POOL = None


register_cache_resetter(_reset_counter_content_pool)


def _counter_content_pool():
    global _COUNTER_CONTENT_POOL
    if _COUNTER_CONTENT_POOL is not None:
        return _COUNTER_CONTENT_POOL
    kb = load_kb(); pool = []; seen_q = set()
    import random as _random
    rng = _random.Random(20240517)     
    contra = lambda x: x[1:] if x.startswith("-") else "-" + x
    PER_REC = 12
    for rec in kb:
        atoms = rec["atoms"]; sup = rec.get("support", []); dis = rec.get("disclaim", [])
        if not sup and not dis:
            continue
        subsets = []
        if sup:
            subsets.append(tuple(sup))
            if len(sup) > 1:
                subsets += [(a,) for a in sup]
        if dis:
            subsets.append(tuple(dis))
            if len(dis) > 1:
                subsets += [(a,) for a in dis]
        made = 0
        for args in subsets:
            if made >= PER_REC:
                break
            ops = _ops_from_args(args, atoms)
            if not ops or sum(1 for o in ops if o.kind in ("defeasible", "strict")) > 10:
                continue
            leaves = [o.content for o in ops if o.kind in ("premise", "axiom")]
            if not leaves:
                continue
            try:
                sm = ASPICVerifier.from_operations(ops, ordering="last_link_elitist").status_map()
            except Exception:
                continue
            cand = []
            for o in ops:
                if o.kind == "defeasible" and sm.get(o.consequent) == JUSTIFIED \
                        and o.consequent not in cand:
                    cand.append(o.consequent)
            rules_disp = [o for o in ops if o.kind in ("defeasible", "strict")]
            idx = {o.name: i + 1 for i, o in enumerate(rules_disp)}
            drive = leaves[0]
            for target in cand:
                if made >= PER_REC:
                    break
                qkey = (_render_content_theory(atoms, ops), target)
                if qkey in seen_q:
                    continue
                top = [o.name for o in ops if o.kind == "defeasible" and o.consequent == target]
                if not top:
                    continue
                top_ops = {o.name: o for o in ops if o.name in top}
                use_pref = True
                added = []
                neg_rule = Operation(kind="defeasible", name="_neg", antecedents=(drive,),
                                     consequent=contra(target))
                added.append(neg_rule)
                outpref = []       
                undercut = []        
                prem_pref = None 
                for j, rn in enumerate(top):
                    roll = rng.random()
                    ants = top_ops[rn].antecedents
                    if use_pref and prem_pref is None and ants and roll < 0.50:
                        p = ants[0]
                        prem_pref = (contra(p), p)
                        added.append(Operation(kind="premise", content=contra(p)))
                        added.append(Operation(kind="prefer_premise", stronger=contra(p), weaker=p))
                        added.append(Operation(kind="defeasible", name=f"_cp{j}",
                                               antecedents=(contra(p),), consequent=contra(target)))
                    elif use_pref and roll < 0.85:
                        outpref.append(rn)
                        added.append(Operation(kind="prefer_rule", stronger="_neg", weaker=rn))
                    else:
                        undercut.append(rn)
                        added.append(Operation(kind="defeasible", name=f"_uc{idx[rn]}",
                                               antecedents=(drive,), consequent=contra(rn)))
                try:
                    v = ASPICVerifier.from_operations(ops, ordering="last_link_elitist")
                    for op in added:
                        v.fw.apply(op)
                    if not (v.status(contra(target)) == JUSTIFIED and v.is_consistent()):
                        continue
                except Exception:
                    continue
                seen_q.add(qkey)
                base_rules = sum(1 for o in ops if o.kind in ("defeasible", "strict"))
                counter_rule_no = base_rules + 1
                wl = []
                wl.append(f"[defeasible: {_gloss(atoms, drive)} => {_gloss(atoms, contra(target))}]")
                for rn in undercut:
                    wl.append(f"[defeasible: {_gloss(atoms, drive)} => -Rule {idx[rn]}]")
                for rn in outpref:
                    wl.append(f"[prefer_rule: Rule {counter_rule_no} > Rule {idx[rn]}]")
                if prem_pref is not None:
                    cp, p = prem_pref
                    wl.append(f"[premise: {_gloss(atoms, cp)}]")
                    wl.append(f"[prefer_premise: {_gloss(atoms, cp)} > {_gloss(atoms, p)}]")
                    wl.append(f"[defeasible: {_gloss(atoms, cp)} => {_gloss(atoms, contra(target))}]")
                pool.append((atoms, ops, target, "\n".join(wl))); made += 1
    _COUNTER_CONTENT_POOL = pool
    return pool


def _counter_content(rng, level=2):
    pool = _counter_content_pool()
    if not pool:
        return None
    return _pick_by_level(pool, level, rng, key=lambda e: _n_rules(e[1]), tag="counter_content")


def frame_counter_argumentation(rng, with_content=False, level=2):
    if with_content:
        built = _counter_content(rng, level)
        if built is None:
            return None
        atoms, ops, T, witness = built
        neg = contrary(T)
        theory_text = _render_content_theory(atoms, ops); notation = _CONTENT_NOTATION; mode = "content"
        tgt, ntgt = _gloss(atoms, T), _gloss(atoms, neg)
        meta_extra = {"atoms": atoms, "n_supports": sum(1 for o in ops if o.kind == "defeasible" and o.consequent == T)}
        ref = witness
    else:
        k = max(1, task_elements("enthymeme", level))
        ops, T, supports, witness = _build_ksupport(rng, k, level)
        neg = contrary(T)
        theory_text = _render_symbolic_theory(ops); notation = _sym_notation(level); mode = "symbolic"
        tgt, ntgt = T, neg
        meta_extra = {"n_supports": k, "supports": supports}
        ref = witness.replace("][", "]\n[")
    task = (notation + "\n\nTheory:\n" + theory_text + f"\n\nThe claim \"{tgt}\" is justified, backed by "
            f"independent line(s) of argument. Construct a counter-case that makes {ntgt} justified "
            f"under grounded semantics. A bare contradiction of \"{tgt}\" only deadlocks with it; to make "
            "the opposite prevail you must contend with each supporting line (undercut its rule, or "
            "out-prefer it). The negation of a statement x is written -x. Use ONLY ordinary premises, "
            "defeasible rules, and preference directives: an added axiom or strict rule is not a "
            "counter-ARGUMENT and scores nothing.")
    if mode == "content":
        how = (" Write your directives one per line: [premise: a statement], "
               "[defeasible: A AND B => C], [strict: A AND B -> C]. Every statement you use, whether "
               "a premise or part of a rule, must be one that already appears in the theory above "
               "(or its negation); you may add new rules linking those statements, but do not "
               "introduce statements of your own. Negate a statement by prefixing "
               "'-', e.g. [premise: -the sky is blue]. To switch off a rule (undercut), negate its "
               "label, e.g. [defeasible: some statement => -Rule 2] or [premise: -Rule 2]. To "
               "out-prefer a line instead, add your own rule for the opposite conclusion with no "
               "label, e.g. [defeasible: A => B]; it becomes the next rule after those shown (so if "
               "Rules 1-N are listed, it is Rule N+1), which you name in [prefer_rule: Rule N+1 > "
               "Rule k].")
    else:
        how = (" Write your directives one per line: [premise: x], [defeasible: a AND b => c], "
               "[strict: a AND b -> c]. To rebut, derive the negation -c of a conclusion; to "
               "undercut rule dK, derive or assert -dK.")
    task = task + how
    spec = ("Plan your counter-case in the reasoning block. The answer block must contain your "
            "directives, one per line, and nothing else; keep the theory consistent.")
    prompt = _intro(level) + "\n\n" + _ordering_decl(GYM_ORDERING) + task + "\n\n" + _format_block(spec)
    return _entry("counter_argumentation", prompt, ref, ops, GYM_ORDERING, target=T, mode=mode, **meta_extra)
