import random

from aspic_gym import (
    GYM_ORDERING, ASPICVerifier, Operation, JUSTIFIED, _atoms, _gloss, _entry, _render_symbolic_theory, _render_content_theory, contrary, ops_to_dicts,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl
import levels as difficulty
from tasks.status_query import _status_kb_theory, _status_symbolic_theory

def _kinds_at(level):
    return difficulty.recipe('perturbation_kinds', level)
_STATUS_TEXT = {
    "JUSTIFIED": "justified", "OVERRULED": "overruled",
    "UNDECIDED": "undecided", "UNSATISFIABLE": "unsatisfiable",
}


def _ante_gloss(g, ante):
    return " and ".join(f'"{g(a)}"' for a in ante) if ante else "the current facts"


def _status_name(v, lit):
    return _STATUS_TEXT[str(v.status(lit))]


def _next_rule_name(ops):
    n = max([int(o.name[1:]) for o in ops
             if o.kind in ("defeasible", "strict") and o.name and o.name[1:].isdigit()] or [0])
    return f"d{n + 1}"


def _build_perturbation(rng, ops, v_pre, kind):
    prem = [o for o in ops if o.kind == "premise"]
    axioms = [o for o in ops if o.kind == "axiom"]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    derived = [o.consequent for o in rules]
    syms = _atoms(ops)

    if kind == "retract_premise":
        if not prem:
            return None
        p = prem[rng.randrange(len(prem))].content
        post = [o for o in ops if not (o.kind == "premise" and o.content == p)]
        return (post, f"the premise {p} is RETRACTED from the theory",
                lambda g: f'the claim "{g(p)}" is RETRACTED from the theory')

    if kind == "add_premise":
        pool = [a for o in rules for a in o.antecedents
                if not a.startswith("-") and str(v_pre.status(a)) != "JUSTIFIED"]
        pool = [a for a in dict.fromkeys(pool) if a in syms or a.lstrip("-") in syms]
        if not pool:
            return None
        p = pool[rng.randrange(len(pool))]
        post = list(ops) + [Operation(kind="premise", content=p)]
        return (post, f"the premise {p} is ADDED to the theory",
                lambda g: f'the claim "{g(p)}" is ADDED to the theory as an ordinary premise')

    if kind == "add_rule":
        heads = [y for y in dict.fromkeys(derived)
                 if str(v_pre.status(y)) == "JUSTIFIED" and not y.lstrip("-").startswith("d")]
        srcs = [x for x in syms if str(v_pre.status(x)) == "JUSTIFIED"]
        rng.shuffle(heads); rng.shuffle(srcs)
        for y in heads:
            for x in srcs:
                if x == y or x == contrary(y):
                    continue
                nm = _next_rule_name(ops)
                new = Operation(kind="defeasible", name=nm, antecedents=(x,),
                                consequent=contrary(y))
                post = list(ops) + [new]
                return (post,
                        f"the rule [defeasible {nm}: {x} => {contrary(y)}] is ADDED",
                        lambda g, x=x, y=y, nm=nm:
                            f'a new defeasible rule is ADDED: "{g(x)}" now gives a reason for '
                            f'"{g(contrary(y))}"')
        return None

    if kind == "remove_rule":
        if not rules:
            return None
        r = rules[rng.randrange(len(rules))]
        post = [o for o in ops if not (o.kind in ("defeasible", "strict") and o.name == r.name)]
        return (post, f"the rule {r.name} is REMOVED from the theory",
                lambda g, r=r: f'the rule concluding "{g(r.consequent)}" '
                               f'(from {_ante_gloss(g, r.antecedents)}) is REMOVED from the theory')

    if kind == "add_pref":
        defs = [o for o in rules if o.kind == "defeasible"]
        pairs = [(a, b) for a in defs for b in defs
                 if a.name != b.name and a.consequent == contrary(b.consequent)]
        existing = {(o.stronger, o.weaker) for o in ops if o.kind == "prefer_rule"}
        pairs = [(a, b) for a, b in pairs
                 if (a.name, b.name) not in existing and (b.name, a.name) not in existing]
        if not pairs:
            return None
        a, b = pairs[rng.randrange(len(pairs))]
        post = list(ops) + [Operation(kind="prefer_rule", stronger=a.name, weaker=b.name)]
        desc = f"the preference [prefer_rule: {a.name} > {b.name}] is ADDED"
        return (post, desc,
                lambda g, a=a, b=b: f'a preference is ADDED: the reason for "{g(a.consequent)}" '
                                    f'now outranks the reason for "{g(b.consequent)}"')

    if kind == "undercut":
        cands = [o for o in rules if o.kind == "defeasible"
                 and str(v_pre.status(o.consequent)) == "JUSTIFIED"]
        if not cands:
            return None
        r = cands[rng.randrange(len(cands))]
        post = list(ops) + [Operation(kind="premise", content="-" + r.name)]
        desc = (f"the premise -{r.name} is ADDED, undercutting rule {r.name} "
                f"(the rule no longer applies)")
        return (post, desc,
                lambda g, r=r: f'a recognized EXCEPTION now blocks the rule that infers '
                               f'"{g(r.consequent)}" from {_ante_gloss(g, r.antecedents)} '
                               f'(that rule no longer applies)')

    if kind == "downgrade_axiom":
        if not axioms:
            return None
        a = axioms[rng.randrange(len(axioms))].content
        post = [Operation(kind="premise", content=o.content)
                if (o.kind == "axiom" and o.content == a) else o for o in ops]
        return (post, f"the axiom {a} is DOWNGRADED to an ordinary (challengeable) premise",
                lambda g: f'the certain fact "{g(a)}" is DOWNGRADED to an ordinary '
                          f'(challengeable) premise')
    return None


def _pred_task_block(theory_text, notation, pre_lines, pert_lines, claims_block, n):
    plural = "s, applied in order" if len(pert_lines) > 1 else ""
    return (
        "You are working with a defeasible argumentation theory. " + notation + "\n\n"
        "Theory:\n" + theory_text + "\n\n"
        "Under grounded semantics (the most cautious evaluation), the CURRENT statuses of the "
        f"{n} statement(s) below are:\n"
        + "\n".join(pre_lines) + "\n\n"
        f"Perturbation{plural}:\n" + "\n".join(pert_lines) + "\n\n"
        "A status is one of:\n"
        "- justified: it has an argument that survives every attack, so it is accepted.\n"
        "- overruled: it has an argument, but a stronger surviving argument defeats it.\n"
        "- undecided: it is caught in an unresolved conflict, neither accepted nor defeated.\n"
        "- unsatisfiable: there is no argument for it at all in the changed theory.")


def frame_perturbation_prediction(rng, with_content=False, level=1, ordering=None):
    n = difficulty.knob('perturbation_prediction', 'n_queries', level)
    ordering = ordering or GYM_ORDERING
    weak = "weakest_link" in ordering
    roll = rng.random()
    want = "none" if roll < 0.25 else ("all" if roll < 0.5 else "some")
    for _ in range(60):
        base_level = min(level, 9)
        if with_content:
            built = _status_kb_theory(rng, base_level, ordering)
            if not built:
                return None
            _rec, atoms, ops = built
            gloss = lambda l: _gloss(atoms, l)
            theory_text = _render_content_theory(atoms, ops)
            notation = _CONTENT_NOTATION
        else:
            ops, _ord = _status_symbolic_theory(rng, base_level, ordering)
            atoms = None
            gloss = lambda l: l
            theory_text = _render_symbolic_theory(ops)
            notation = _sym_notation(level)
        v_pre = ASPICVerifier.from_operations(ops, ordering=ordering)

        n_pert = difficulty.knob('perturbation_prediction', 'n_perturb', level)
        kinds = _kinds_at(level)
        post_ops, sym_descs, con_descs = list(ops), [], []
        ok = True
        for _p in range(n_pert):
            v_now = ASPICVerifier.from_operations(post_ops, ordering=ordering)
            built_p = None
            for k in rng.sample(kinds, len(kinds)):
                built_p = _build_perturbation(rng, post_ops, v_now, k)
                if built_p:
                    break
            if not built_p:
                ok = False
                break
            post_ops, sd, cd = built_p
            sym_descs.append(sd)
            con_descs.append(cd(gloss))
        if not ok:
            continue
        v_post = ASPICVerifier.from_operations(post_ops, ordering=ordering)

        syms = _atoms(ops)
        lits = syms + [contrary(s) for s in syms]
        lits = [l for l in lits if not l.lstrip("-").startswith("d")]
        pre = {l: _status_name(v_pre, l) for l in lits}
        post = {l: _status_name(v_post, l) for l in lits}
        changed = [l for l in lits if pre[l] != post[l]]
        unchanged = [l for l in lits if pre[l] == post[l]]

        rng.shuffle(changed); rng.shuffle(unchanged)
        if want == "none":
            if len(unchanged) < n:
                continue
            picked = unchanged[:n]
        elif want == "all":
            if len(changed) < n:
                continue
            picked = changed[:n]
        else: 
            n_ch = rng.randint(1, max(1, n - 1))
            if len(changed) < n_ch or len(unchanged) < n - n_ch:
                continue
            picked = changed[:n_ch] + unchanged[:n - n_ch]
        rng.shuffle(picked)

        changed_idx = [(i, post[l]) for i, l in enumerate(picked) if pre[l] != post[l]]

        pre_lines = [f"  {i + 1}. {gloss(l)} -- currently {pre[l]}" for i, l in enumerate(picked)]
        pert_lines = [f"  {'first, ' if n_pert > 1 and i == 0 else 'then, ' if n_pert > 1 else ''}"
                      f"{(con_descs if with_content else sym_descs)[i]}"
                      for i in range(n_pert)]
        order_decl = _ordering_decl(ordering)
        spec = (
            "After the change, some statuses may change and others may not. In the answer, list "
            "ONLY the statements whose status CHANGED, one per line, as `number: new_status` "
            "(e.g. `2: overruled`). If NO listed statement changed, answer exactly `none`. "
            "Do not list statements whose status stayed the same.")
        prompt = (_intro(level) + "\n\n" + order_decl
                  + _pred_task_block(theory_text, notation, pre_lines, pert_lines, "", n)
                  + "\n\n" + _format_block(spec))
        if changed_idx:
            ref = "\n".join(f"{i + 1}: {g}" for i, g in changed_idx)
        else:
            ref = "none"
        meta = dict(queries=picked,
                    pre_statuses=[pre[l] for l in picked],
                    post_statuses=[post[l] for l in picked],
                    changed=[i for i, _g in changed_idx],
                    changed_gold={i: g for i, g in changed_idx},
                    change_type=want,
                    post_ops=ops_to_dicts(post_ops),
                    perturbation=sym_descs, n_claims=n,
                    mode="content" if with_content else "symbolic")
        if with_content:
            meta["atoms"] = atoms
        return _entry("perturbation_prediction", prompt, ref, ops, ordering, **meta)
    return None
