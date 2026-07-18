
from aspic_gym import (
    GYM_ORDERING, enth_components, Operation, ASPICVerifier, JUSTIFIED, contrary, render_dsl, load_kb, task_elements, _fresh_syms, _rename_in_order, _sentence_dsl, _gloss, _entry, _render_symbolic_theory, _render_content_theory,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl


def _nonadjacent(rng, lo, hi, count):
    avail = list(range(lo, hi + 1)); rng.shuffle(avail)
    chosen = []
    for p in avail:
        if all(abs(p - c) > 1 for c in chosen):
            chosen.append(p)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_enthymeme_chain(rng, k, level=2):
    m = 2 * k                                  
    syms = _fresh_syms(rng, m + 1)
    use_neg = level >= 4
    pol = [""] * (m + 1)
    for i in range(1, m + 1):
        pol[i] = "-" if (use_neg and rng.random() < 0.3) else ""
    lit = lambda i: pol[i] + syms[i]
    T = lit(m)
    spine = [Operation(kind="premise", content=syms[0])]
    for i in range(1, m + 1):
        st = rng.random() < 0.25
        spine.append(Operation(kind="strict" if st else "defeasible",
                               name=f"{'s' if st else 'd'}{i}",
                               antecedents=(lit(i - 1),), consequent=lit(i)))
    rem = _nonadjacent(rng, 0, m, k)
    if rem is None:
        return None
    removed = [spine[p] for p in rem]
    kept = [spine[i] for i in range(len(spine)) if i not in set(rem)]
    rng.shuffle(kept)                          
    kept, _ = _rename_in_order(kept)
    return kept, removed, T


def _build_enthymeme_pref(rng, k):
    m = 2 * k
    syms = _fresh_syms(rng, m + 1)
    T = syms[m]
    spine = [Operation(kind="premise", content=syms[0])]
    for i in range(1, m + 1):
        spine.append(Operation(kind="defeasible", name=f"d{i}",
                               antecedents=(syms[i - 1],), consequent=syms[i]))
    last_rule = spine[m]
    premise_tie = rng.random() < 0.5
    if premise_tie:
        rebut = [Operation(kind="premise", content="-" + syms[0])]
    else:
        hsym = _fresh_syms(rng, 1)[0]
        while hsym in syms:
            hsym = hsym + "0"
        rebut = [Operation(kind="premise", content=hsym),
                 Operation(kind="defeasible", name="dr", antecedents=(hsym,), consequent="-" + T)]
    gaps = _nonadjacent(rng, 1, m - 1, k - 1) if k > 1 else []
    if k > 1 and gaps is None:
        return None
    gaps = set(gaps or [])
    removed_chain = [spine[p] for p in gaps]
    kept = [spine[i] for i in range(len(spine)) if i not in gaps] + rebut
    rng.shuffle(kept)                      
    kept, name_map = _rename_in_order(kept)
    ndef = sum(1 for o in kept if o.kind == "defeasible")
    removed_chain = [Operation(kind="defeasible", name=f"d{ndef + 1 + i}",
                               antecedents=r.antecedents, consequent=r.consequent)
                     for i, r in enumerate(removed_chain)]
    if premise_tie:
        pref = Operation(kind="prefer_premise", stronger=syms[0], weaker="-" + syms[0])
    else:
        pref = Operation(kind="prefer_rule", stronger=name_map[last_rule.name], weaker=name_map["dr"])
    removed = removed_chain + [pref]
    return kept, removed, T


def _enth_content(rng, k):
    want_pref = rng.random() < 0.35
    got = _enth_content_pass(rng, k, want_pref)
    if got is None and want_pref:
        got = _enth_content_pass(rng, k, False)
    return got


def _enth_content_pass(rng, k, want_pref):
    kb = load_kb()
    recs = [r for r in kb if r.get("support")]
    rng.shuffle(recs)
    for rec in recs:
        atoms = rec["atoms"]
        sup = sorted(rec.get("support", []),
                     key=lambda a: -(len(a["premises"]) + len(a.get("axioms", [])) + len(a["rules"])))
        dis = sorted(rec.get("disclaim", []),
                     key=lambda a: -(len(a["premises"]) + len(a.get("axioms", [])) + len(a["rules"])))
        args = ([(a, "disclaim") for a in dis] if rng.random() < 0.6 else []) \
            + [(a, "support") for a in sup] + [(a, "disclaim") for a in dis]
        seen_args = set()
        for arg, stance in args:
            aid = id(arg)
            if aid in seen_args:
                continue
            seen_args.add(aid)
            target_lit = "c0" if stance == "support" else "-c0"
            full = []
            for lit in arg["premises"]:
                full.append(Operation(kind="premise", content=lit))
            for lit in arg.get("axioms", []):
                kk = "axiom" if atoms.get(lit.lstrip("-"), {}).get("axiomatic") else "premise"
                full.append(Operation(kind=kk, content=lit))
            for r in arg["rules"]:
                kk = "strict" if r.get("strict") else "defeasible"
                full.append(Operation(kind=kk, name=r["name"], antecedents=tuple(r["antecedents"]),
                                      consequent=r["consequent"]))
            removable = [o for o in full if o.kind in ("premise", "axiom", "defeasible", "strict")]
            keep_floor = max(3, (len(removable) + 1) // 2)  
            k_eff = min(k, len(removable) - keep_floor)
            if k_eff < 1:
                continue
            try:
                if ASPICVerifier.from_operations(full, ordering="last_link_elitist").status(target_lit) != JUSTIFIED:
                    continue
            except Exception:
                continue
            removed = rng.sample(removable, k_eff)
            if want_pref:
                ordp = [o.content for o in full if o.kind == "premise"]
                if ordp:
                    p = rng.choice(ordp)
                    challenger = Operation(kind="premise", content=contrary(p))
                    pref = Operation(kind="prefer_premise", stronger=p, weaker=contrary(p))
                    chain_rm = rng.sample(removable, k_eff - 1) if k_eff > 1 else []
                    kept2 = [o for o in full if o not in chain_rm] + [challenger]
                    removed2 = chain_rm + [pref]
                    try:
                        v0 = ASPICVerifier.from_operations(kept2, ordering="last_link_elitist")
                        if v0.status(target_lit) != JUSTIFIED:
                            v1 = ASPICVerifier.from_operations(kept2, ordering="last_link_elitist")
                            for op in removed2:
                                v1.fw.apply(op)
                            try:
                                alone = ASPICVerifier.from_operations(
                                    removed2, ordering="last_link_elitist").status(target_lit) == JUSTIFIED
                            except Exception:
                                alone = False
                            if (v1.status(target_lit) == JUSTIFIED and v1.is_consistent()
                                    and not alone):
                                return atoms, kept2, removed2, target_lit
                    except Exception:
                        pass
                continue                              
            kept = [o for o in full if o not in removed]
            try:
                if ASPICVerifier.from_operations(kept, ordering="last_link_elitist").status(target_lit) == JUSTIFIED:
                    continue                    
                if ASPICVerifier.from_operations(removed, ordering="last_link_elitist").status(target_lit) == JUSTIFIED:
                    continue                        
            except Exception:
                continue
            return atoms, kept, removed, target_lit
    return None


def frame_enthymeme(rng, with_content=False, level=2):
    k = max(1, task_elements("enthymeme", level))  
    if with_content:
        built = _enth_content(rng, k)
        if built is None:
            return None
        atoms, kept, removed, T = built
        gl = {a: {"pos": atoms[a]["pos"], "neg": atoms[a]["neg"]} for a in atoms}
        theory_text = _render_content_theory(atoms, kept); notation = _CONTENT_NOTATION; mode = "content"
        gold = _sentence_dsl(removed, gl); tgt = _gloss(atoms, T)
        k = len(removed)                       
        meta_extra = {"atoms": atoms}
    else:
        built = None
        if k >= 1 and rng.random() < 0.35:
            built = _build_enthymeme_pref(rng, k)
        if built is None:
            built = _build_enthymeme_chain(rng, k, level)
        if built is None:
            return None
        kept, removed, T = built
        theory_text = _render_symbolic_theory(kept); notation = _sym_notation(level); mode = "symbolic"
        gold = render_dsl(removed); tgt = T
        meta_extra = {}
    task = (notation + "\n\nTheory:\n" + theory_text + f"\n\nThe claim \"{tgt}\" cannot yet be "
            f"justified: the argument for it is missing exactly {k} element(s) - these may be "
            "premises, rules (of either kind), or a preference (between rules, or between "
            "premises). Supply the missing "
            f"element(s) so that \"{tgt}\" becomes justified under grounded semantics. The negation of a "
            "statement x is written -x.")
    if with_content:
        spec = ("Work out what is missing before answering: trace the argument from its premises "
                "toward the target, find where the derivation breaks, and supply exactly the missing "
                "element(s). The answer block must contain the missing directives, one per line, and "
                "nothing else; you are graded on restoring the intended argument, not on inventing a "
                "new shortcut to the conclusion.")
    else:
        spec = (f"Trace the argument from its premises toward the target, find where the "
                f"derivation breaks, and supply exactly {k} element(s) that complete the existing "
                f"chain so the target becomes justified. The answer block must contain exactly "
                f"{k} directive(s), one per line, and nothing else. Each element you add must be "
                f"used in the completed argument, and must build on the given theory rather than "
                f"assert a separate argument for the target.")
    prompt = _intro(level) + "\n\n" + _ordering_decl(GYM_ORDERING) + task + "\n\n" + _format_block(spec)
    ref = "[answer]\n" + gold + "\n[/answer]"
    return _entry("enthymeme", prompt, ref, kept, GYM_ORDERING, target=T, mode=mode,
                  n_missing=k, gold_components=sorted(enth_components(removed)), **meta_extra)
