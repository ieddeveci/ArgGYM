
from aspic_gym import (
    GYM_ORDERING, Operation, ASPICVerifier, JUSTIFIED, _kb_status_theory, _gloss, _entry, _render_symbolic_theory, _render_content_theory, _RULE_NAME,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl
import levels as difficulty


def _claim_id_symbolic(rng, level, ordering=None):
    ordering = ordering or GYM_ORDERING
    weak = "weakest_link" in ordering
    nchains = difficulty.knob('claim_identification', 'n_chains', level)
    depth = difficulty.knob('claim_identification', 'depth', level)
    low_variety = level <= 2
    if low_variety:
        nchains = rng.choice([1, 2])
        depth = rng.choice([1, 2])
    allow_neg = difficulty.gate('claim_identification', 'allow_neg', level)
    allow_ax = difficulty.gate('claim_identification', 'allow_ax', level)
    allow_strict = difficulty.gate('claim_identification', 'allow_strict', level)
    allow_def = difficulty.gate('claim_identification', 'defeater', level)
    ops = []
    conclusions = []
    chain_roots = []          
    aid = [0]
    di = si = 0
    strict_used = False

    def fresh():
        s = f"a{aid[0]}"; aid[0] += 1; return s

    for ci in range(nchains):
        leaf = fresh()
        ops.append(Operation(kind="axiom" if (allow_ax and ci == 0) else "premise", content=leaf))
        cur = leaf
        for step in range(depth):
            nxt = fresh()
            neg = allow_neg and step == depth - 1 and rng.random() < 0.4
            cons = ("-" + nxt) if neg else nxt
            if allow_strict and not strict_used and ci == 0 and step == 0:
                strict_used = True; si += 1
                ops.append(Operation(kind="strict", name=f"s{si}", antecedents=(cur,), consequent=cons))
            else:
                di += 1
                ops.append(Operation(kind="defeasible", name=f"d{di}", antecedents=(cur,), consequent=cons))
            cur = cons
        conclusions.append(cur)
        chain_roots.append((cur, leaf, allow_ax and ci == 0))
    if low_variety and rng.random() < 0.5:
        distract = fresh()
        ops.append(Operation(kind="premise", content=distract))
        conclusions.append(distract)
    if allow_def and conclusions:
        cand_chains = [(c, r, ax) for (c, r, ax) in chain_roots
                       if next((o for o in ops if o.kind == "defeasible" and o.consequent == c), None)]
        prem_rooted = [(c, r, ax) for (c, r, ax) in cand_chains if not ax]
        chosen = prem_rooted[0] if prem_rooted else (cand_chains[0] if cand_chains else None)
        if chosen:
            C, _root, _ax = chosen
            opp = C[1:] if C.startswith("-") else "-" + C
            target = next((o.name for o in ops if o.kind == "defeasible" and o.consequent == C), None)
        else:
            target = None
        if target:
            leaf2 = fresh(); ops.append(Operation(kind="premise", content=leaf2))
            di += 1; dn = f"d{di}"
            ops.append(Operation(kind="defeasible", name=dn, antecedents=(leaf2,), consequent=opp))
            prem_set = {o.content for o in ops if o.kind == "premise"}
            cons_map = {o.consequent: o for o in ops if o.kind in ("defeasible", "strict")}
            seen, stack, tgt_prems, tgt_rules = set(), [C], set(), set()
            while stack:
                lit = stack.pop()
                if lit in seen:
                    continue
                seen.add(lit)
                if lit in prem_set:
                    tgt_prems.add(lit)
                sub = cons_map.get(lit)
                if sub is not None:
                    if sub.kind == "defeasible" and sub.name:
                        tgt_rules.add(sub.name)
                    stack.extend(sub.antecedents or ())
            if weak:
                for tr in sorted(tgt_rules):
                    ops.append(Operation(kind="prefer_rule", stronger=dn, weaker=tr))
            else:
                ops.append(Operation(kind="prefer_rule", stronger=dn, weaker=target))
            for a in sorted(tgt_prems):
                if a != leaf2:
                    ops.append(Operation(kind="prefer_premise", stronger=leaf2, weaker=a))
    return ops, ordering


def _claim_task_block(theory_text, notation, claims_block, n):
    return ("You are analysing a defeasible argumentation theory. " + notation + "\n\nTheory:\n"
            + theory_text + "\n\nTrace the argument(s) under grounded semantics and decide which of "
            f"the {n} numbered claims below the argument actually ESTABLISHES - that is, which are "
            "JUSTIFIED (derived and not defeated). A claim that is overruled, undecided, or never "
            "derived is NOT established.\n\nClaims:\n" + claims_block)


def frame_claim_identification(rng, with_content=False, level=2, ordering=None):
    if with_content:
        built = _kb_status_theory(rng, level, ordering)
        if not built:
            return None
        rec, atoms, ops = built
        ordering = ordering or GYM_ORDERING
        gloss_fn = lambda l: _gloss(atoms, l)
        theory_text = _render_content_theory(atoms, ops)
        notation = _CONTENT_NOTATION; mode = "content"
    else:
        ops, ordering = _claim_id_symbolic(rng, level, ordering)
        gloss_fn = lambda l: l
        theory_text = _render_symbolic_theory(ops)
        notation = _sym_notation(level); mode = "symbolic"
    v = ASPICVerifier.from_operations(ops, ordering=ordering)
    sm = v.status_map()
    contra = lambda l: l[1:] if l.startswith("-") else "-" + l
    derived = []
    for o in ops:
        # An undercutter concludes a rule name (-d3), which is not a claim the
        # question can ask about: content mode displays rules as "Rule k", so a
        # candidate like "-d3" would name something the prompt never defines.
        if (o.kind in ("defeasible", "strict") and o.consequent not in derived
                and not _RULE_NAME.match(o.consequent)):
            derived.append(o.consequent)
    cand = []
    for d in derived:
        for x in (d, contra(d)):
            if x not in cand:
                cand.append(x)
    in_theory = {(_b := lambda l: l[1:] if l.startswith("-") else l)(x) for x in cand}
    for o in ops:
        if o.kind in ("premise", "axiom"):
            in_theory.add(o.content.lstrip("-"))
        else:
            in_theory.update(a.lstrip("-") for a in o.antecedents)
    theory_pos = set()
    for o in ops:
        if o.kind in ("premise", "axiom"):
            theory_pos.add(o.content.lstrip("-"))
        elif o.kind in ("defeasible", "strict"):
            theory_pos.update(a.lstrip("-") for a in o.antecedents)
            theory_pos.add(o.consequent.lstrip("-"))
    # The distractor is drawn from theory_pos, so rule names have to be excluded
    # here too -- an undercutter's consequent would otherwise reach the claim
    # list by this route even though `derived` already filters them.
    theory_pos = {x for x in theory_pos if not _RULE_NAME.match(x)}
    in_theory = {x for x in in_theory if not _RULE_NAME.match(x)}
    unjust_pos = sorted(x for x in theory_pos
                        if x not in cand and sm.get(x) != JUSTIFIED)
    if unjust_pos:
        fresh = rng.choice(unjust_pos)
    elif with_content:
        fresh = next((a for a in atoms if a not in in_theory and a != "c0"), None)
    else:
        i = 0
        while f"a{i}" in in_theory:
            i += 1
        fresh = f"a{i}"
    if fresh:
        cand.append(fresh)
    rng.shuffle(cand)
    cap = max(2, 2 * level)
    if len(cand) > cap:
        est_pool = [c for c in cand if sm.get(c) == JUSTIFIED]
        non_pool = [c for c in cand if sm.get(c) != JUSTIFIED]
        rng.shuffle(est_pool); rng.shuffle(non_pool)
        half = cap // 2
        take_est = min(len(est_pool), max(1, half))
        take_non = min(len(non_pool), cap - take_est)
        take_est = min(len(est_pool), cap - take_non)     
        cand = est_pool[:take_est] + non_pool[:take_non]
        rng.shuffle(cand)
    established = [i + 1 for i, c in enumerate(cand) if sm.get(c) == JUSTIFIED]
    if not established or len(established) == len(cand):
        return None
    positives = [i + 1 for i, c in enumerate(cand) if not c.startswith("-")]
    if positives == established:
        return None                           
    claims_block = "\n".join(f"  {i + 1}. {gloss_fn(c)}" for i, c in enumerate(cand))
    spec = ("In the answer, list the NUMBERS of every claim the argument establishes, separated by "
            "commas (for example `1, 3`). A claim the argument never even mentions is not "
            "established. Write `none` if it establishes none of them.")
    order_decl = _ordering_decl(ordering or "last_link_elitist")
    prompt = (_intro(level) + "\n\n" + order_decl
              + _claim_task_block(theory_text, notation, claims_block, len(cand))
              + "\n\n" + _format_block(spec))
    ref = "[answer]\n" + (", ".join(map(str, established)) if established else "none") + "\n[/answer]"
    return _entry("claim_identification", prompt, ref, ops, ordering, established=established,
                  universe=cand, n_candidates=len(cand), mode=mode)
