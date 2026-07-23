
from dataclasses import replace

from aspic_gym import (
    GYM_ORDERING, _rename_in_order, Operation, ASPICVerifier, load_kb, _rule_index, _content_lit, _entry, _render_symbolic_theory, _render_content_theory,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl
import levels as difficulty


def _attackers_symbolic(rng, level):
    depth = difficulty.knob('attackers_of', 'depth', level)
    natt = difficulty.knob('attackers_of', 'n_attackers', level)
    pool = difficulty.recipe('attackers_of_pool', level)
    names = [f"{c}{i}" for c in "pqrstuvwxyz" for i in range(10)]
    rng.shuffle(names)
    nm_iter = iter(names)
    fr = lambda: next(nm_iter)
    rlabels = [f"d{i}" for i in range(1, 40)]
    rng.shuffle(rlabels)
    rl_iter = iter(rlabels)
    contra = lambda x: x[1:] if x.startswith("-") else "-" + x
    ops = []
    leaf = fr(); ops.append(Operation(kind="premise", content=leaf))
    cur = leaf; rules = []
    for _ in range(depth):
        nxt = fr(); nm = next(rl_iter)
        ops.append(Operation(kind="defeasible", name=nm, antecedents=(cur,), consequent=nxt))
        rules.append(nm); cur = nxt
    T = cur
    if natt >= len(pool):
        chosen = list(pool) + [rng.choice(pool) for _ in range(natt - len(pool))]
    else:
        chosen = rng.sample(pool, natt)
    rng.shuffle(chosen)
    used_rule = set()
    attacks = []
    for t in chosen:
        if t == "rebut":
            attacks.append(contra(T))
        elif t == "undermine":
            attacks.append(contra(leaf))
        elif t == "undercut" and rules:
            avail = [r for r in rules if r not in used_rule] or rules
            r = rng.choice(avail); used_rule.add(r); attacks.append(contra(r))
    attacks = list(dict.fromkeys(attacks))            
    _underm = contra(leaf)
    _derivable = [i for i, l in enumerate(attacks) if l != _underm]
    derived_idx = rng.choice(_derivable) if (difficulty.gate('attackers_of', 'derived_attacker', level) and _derivable) else -1
    for i, lit in enumerate(attacks):
        if i == derived_idx:                         
            src = fr()
            ops.append(Operation(kind="premise", content=src))
            ops.append(Operation(kind="defeasible", name=next(rl_iter),
                                 antecedents=(src,), consequent=lit))
        else:
            ops.append(Operation(kind="premise", content=lit))
    ops.append(Operation(kind="premise", content=contra(fr())))
    if difficulty.gate('attackers_of', 'second_decoy', level):
        ops.append(Operation(kind="premise", content=contra(fr())))
    if level >= 3:
        attack_lits = set(attacks) | {_underm}
        np1, np2 = fr(), fr()
        ops.append(Operation(kind="premise", content=np1))
        ops.append(Operation(kind="premise", content=np2))
        ops.append(Operation(kind="prefer_premise", stronger=np1, weaker=np2))
        safe_rules = [o.name for o in ops if o.kind == "defeasible"
                      and o.consequent not in attack_lits]
        if len(safe_rules) >= 2:
            a, b = rng.sample(safe_rules, 2)
            ops.append(Operation(kind="prefer_rule", stronger=a, weaker=b))
    rng.shuffle(ops)
    ops, name_map = _rename_in_order(ops)
    remap = lambda l: ("-" + name_map[l[1:]]) if (l.startswith("-") and l[1:] in name_map) else (name_map.get(l, l))
    def _remap_op(o):
        if o.kind in ("premise", "axiom"):
            return replace(o, content=remap(o.content))
        if o.kind in ("defeasible", "strict"):
            return replace(o, antecedents=tuple(remap(a) for a in o.antecedents),
                           consequent=remap(o.consequent))
        if o.kind in ("prefer_rule", "prefer_premise"):
            return replace(o, stronger=remap(o.stronger), weaker=remap(o.weaker))
        return o
    ops = [_remap_op(o) for o in ops]
    return ops, T, GYM_ORDERING


def _attackers_content(rng, level):
    kb = load_kb()
    recs = [r for r in kb if r.get("support") and r.get("disclaim")]
    rng.shuffle(recs)
    natt = difficulty.knob('attackers_of', 'n_attackers', level)
    allow_ax = difficulty.gate('attackers_of', 'allow_ax', level); allow_strict = difficulty.gate('attackers_of', 'allow_strict', level)
    pool = difficulty.recipe('attackers_of_pool', level)
    for rec in recs:
        atoms = rec["atoms"]
        kbatks = [a for a in rec.get("attacks", [])
                  if a.get("type") in pool and a.get("target_stance") == "support"
                  and isinstance(a.get("target_index"), int)
                  and a["target_index"] < len(rec["support"])]
        rng.shuffle(kbatks)
        if kbatks:
            ti = kbatks[0]["target_index"]
            tgt = rec["support"][ti]
            inject = [a for a in kbatks if a["target_index"] == ti][:max(1, natt - 1)]
        else:
            tgt = rng.choice(rec["support"])
            inject = []
        atks = rec["disclaim"][:]; rng.shuffle(atks)
        atks = atks[:max(1, natt - len(inject))]
        ops = []; seen_f = set(); seen_r = set()
        contra = lambda x: x[1:] if x.startswith("-") else "-" + x

        def add_arg(arg):
            for lit in arg["premises"]:
                if lit in seen_f or contra(lit) in seen_f:
                    continue
                seen_f.add(lit); ops.append(Operation(kind="premise", content=lit))
            for lit in arg.get("axioms", []):
                if lit in seen_f or contra(lit) in seen_f:
                    continue
                seen_f.add(lit)
                k = "axiom" if (allow_ax and atoms.get(lit.lstrip("-"), {}).get("axiomatic")) else "premise"
                ops.append(Operation(kind=k, content=lit))
            for r in arg["rules"]:
                key = (tuple(r["antecedents"]), r["consequent"])
                if key in seen_r:
                    continue
                seen_r.add(key)
                k = "strict" if (allow_strict and r.get("strict")) else "defeasible"
                ops.append(Operation(kind=k, name=r["name"], antecedents=tuple(r["antecedents"]),
                                     consequent=r["consequent"]))
        add_arg(tgt)
        for a in atks:
            add_arg(a)
        for a in inject:
            for lit in a.get("premises", []):
                if lit in seen_f:
                    continue
                seen_f.add(lit); ops.append(Operation(kind="premise", content=lit))
            for r in a.get("rules", []):
                key = (tuple(r["antecedents"]), r["consequent"])
                if key in seen_r:
                    continue
                seen_r.add(key)
                ops.append(Operation(kind="defeasible", name=r.get("name"),
                                     antecedents=tuple(r["antecedents"]),
                                     consequent=r["consequent"]))
        if level >= 3:
            used = {o.content for o in ops if o.kind in ("premise", "axiom")}
            used |= {o.consequent for o in ops if o.kind in ("defeasible", "strict")}
            tconc = tgt["conclusion"]
            attack_lits = {tconc, contra(tconc)}
            spare = [aid for aid in atoms
                     if atoms[aid].get("pos") and aid not in used and aid != "c0"
                     and aid not in attack_lits and contra(aid) not in used][:2]
            if len(spare) == 2:
                ops.append(Operation(kind="premise", content=spare[0]))
                ops.append(Operation(kind="premise", content=spare[1]))
                ops.append(Operation(kind="prefer_premise", stronger=spare[0], weaker=spare[1]))
            safe_rules = [o.name for o in ops if o.kind == "defeasible" and o.name
                          and o.consequent not in attack_lits]
            if len(safe_rules) >= 2:
                a, b = rng.sample(safe_rules, 2)
                ops.append(Operation(kind="prefer_rule", stronger=a, weaker=b))
        return atoms, ops, tgt["conclusion"]
    return None


def frame_attackers_of(rng, with_content=False, level=2):
    if with_content:
        built = _attackers_content(rng, level)
        if not built:
            return None
        atoms, ops, T = built
        ordering = GYM_ORDERING; _ri = _rule_index(ops); gloss = lambda l: _content_lit(atoms, l, _ri)
        theory_text = _render_content_theory(atoms, ops); notation = _CONTENT_NOTATION; mode = "content"
    else:
        ops, T, ordering = _attackers_symbolic(rng, level)
        gloss = lambda l: l
        theory_text = _render_symbolic_theory(ops); notation = _sym_notation(level); mode = "symbolic"
    v = ASPICVerifier.from_operations(ops, ordering=ordering)
    ag = v.attack_graph()
    concl = {a["id"]: a["conclusion"] for a in ag["arguments"]}
    tids = {a["id"] for a in ag["arguments"] if a["conclusion"] == T}
    if not tids:
        return None
    gold_lits = sorted({concl[e["from"]] for e in ag["defeats"] if e["to"] in tids})
    if not gold_lits:
        return None
    task = ("You are analysing a defeasible argumentation theory. " + notation + "\n\nTheory:\n"
            + theory_text + f"\n\nConsider the argument whose conclusion is:  \"{gloss(T)}\"\n\n"
            "Find every DIRECT attacker of that argument - each argument in the theory that defeats "
            "it by contradicting its conclusion, undermining a premise it relies on, or undercutting "
            "one of its rules.")
    if mode == "symbolic":
        spec = ("In the answer, write the CONCLUSION of every attacker - the exact literal it "
                "derives. For an attack that switches off a rule (an undercut), write the negated "
                "rule label, e.g. -d2. Separate them with commas; write `none` if nothing attacks it.")
        ref = ", ".join(gold_lits)
        prompt = _intro(level) + "\n\n" + _ordering_decl(ordering) + task + "\n\n" + _format_block(spec)
        return _entry("attackers_of", prompt, ref, ops, ordering, mode=mode, target=T,
                      gold_lits=gold_lits)
    gold_sents = [gloss(l) for l in gold_lits]
    distr = sorted({concl[i] for i in concl if i not in tids and concl[i] not in gold_lits and concl[i] != T})
    distractor_sents = [gloss(l) for l in distr]
    spec = ("In the answer, write out the exact statement that each attacker concludes (the claim "
            "that contradicts the argument or one of its premises), one per line. For an attacker "
            "that switches off a rule (an undercut), write the negated rule label, e.g. -Rule 2. "
            "Write `none` if nothing attacks it.")
    ref = "\n".join(gold_sents)
    prompt = _intro(level) + "\n\n" + _ordering_decl(ordering) + task + "\n\n" + _format_block(spec)
    return _entry("attackers_of", prompt, ref, ops, ordering, mode=mode, target=T,
                  gold_lits=gold_lits, gold_sents=gold_sents, distractor_sents=distractor_sents)
