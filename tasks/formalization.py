from aspic_gym import (
    GYM_ORDERING, Operation, parse_dsl, render_dsl, load_kb, _ev_base, _kb_gloss_pool, _components, _sentence_dsl, _entry, _render_content_theory, _rule_index, _relabel_rule_refs, _content as _content_mod,
)
from prompting import _format_block
import levels as difficulty


def _kb_formalization_theory(rng, n):
    kb = load_kb()
    if not kb:
        return None
    recs = kb[:]
    rng.shuffle(recs)
    all_cands = []                      
    for rec in recs:
        atoms = rec["atoms"]
        for s in ("support", "disclaim"):
            for arg in rec[s]:
                rules = arg["rules"]
                for i in range(len(rules)):
                    for k in range(1, len(rules) - i + 1):
                        sub = rules[i:i + k]
                        produced = {r["consequent"] for r in sub}
                        used = [x for r in sub for x in r["antecedents"]]
                        leaves = [x for x in dict.fromkeys(used) if x not in produced]
                        contra = lambda p: p[1:] if p.startswith("-") else "-" + p
                        if any(contra(x) in leaves for x in leaves):
                            continue
                        has_neg = any(r["consequent"].startswith("-") for r in sub)
                        all_cands.append((len(leaves) + len(sub), leaves, sub, atoms, has_neg))
    if not all_cands:
        return None
    exact_neg = [c for c in all_cands if c[4] and c[0] == n]
    near_neg = sorted([c for c in all_cands if c[4]], key=lambda c: abs(c[0] - n))
    exact_any = [c for c in all_cands if c[0] == n]
    if exact_neg:
        pick = rng.choice(exact_neg)
    elif near_neg and abs(near_neg[0][0] - n) <= 2 and rng.random() < 0.8:
        best_off = abs(near_neg[0][0] - n)
        pick = rng.choice([c for c in near_neg if abs(c[0] - n) == best_off])
    elif exact_any:
        pick = rng.choice(exact_any)
    else:
        pick = min(all_cands, key=lambda c: abs(c[0] - n))
    _, leaves, sub, atoms, _ = pick
    return atoms, leaves, sub
    return None

def FORM_COMPOSITION_AT(level):
    k = lambda n: difficulty.knob('formalization', n, level)
    return dict(prem=k('prem'), ax=k('ax'), rules=k('rules'), strict=k('strict'),
                neg=difficulty.gate('formalization', 'neg', level), arity=k('arity'))


def _formalization_glosses(rng, n, content, start=0):
    syms = [f"a{start + i}" for i in range(n)]
    if content:
        kb = load_kb()
        rec = rng.choice(kb) if kb else None
        pairs = [(a["pos"], a["neg"]) for i, a in (rec["atoms"].items() if rec else [])
                 if i != "c0" and a.get("pos") and a.get("neg")] if rec else []
        rng.shuffle(pairs)
        if len(pairs) >= n:
            return {s: {"pos": p, "neg": q} for s, (p, q) in zip(syms, pairs)}
        pool = _kb_gloss_pool()[:] or _content_mod.PROPOSITION_POOL[:]  
        rng.shuffle(pool)
        return {s: {"pos": pool[i % len(pool)],
                    "neg": "it is not the case that " + pool[i % len(pool)]}
                for i, s in enumerate(syms)}
    tags = [chr(65 + i) for i in range(26)]
    rng.shuffle(tags)
    return {s: {"pos": f"claim {tags[i % len(tags)]}",
                "neg": f"it is not the case that claim {tags[i % len(tags)]}"}
            for i, s in enumerate(syms)}


def _formalization_theory(rng, level, content):
    c = FORM_COMPOSITION_AT(level)
    n_facts = c["prem"] + c["ax"]
    gl = _formalization_glosses(rng, n_facts + c["rules"], content)
    syms = list(gl)
    ops, facts = [], syms[:n_facts]
    for i, f in enumerate(facts):
        ops.append(Operation(kind="axiom" if i < c["ax"] else "premise", content=f))
    derivable, heads = list(facts), syms[n_facts:]
    di = si = 0
    for r in range(c["rules"]):
        head = heads[r]
        k = rng.randint(1, min(c["arity"], len(derivable)))
        ants = tuple(rng.sample(derivable, k))
        is_strict = r < c["strict"]
        negate = c["neg"] and (not is_strict) and rng.random() < 0.5
        cons = ("-" + head) if negate else head
        if is_strict:
            si += 1
            ops.append(Operation(kind="strict", name=f"s{si}", antecedents=ants, consequent=cons))
        else:
            di += 1
            ops.append(Operation(kind="defeasible", name=f"d{di}", antecedents=ants, consequent=cons))
        if not negate:
            derivable.append(head)
    if level >= 3:
        drules = [o.name for o in ops if o.kind == "defeasible"]
        oprems = [o.content for o in ops if o.kind == "premise"]
        if len(drules) >= 2:
            a, b = rng.sample(drules, 2)
            ops.append(Operation(kind="prefer_rule", stronger=a, weaker=b))
        if len(oprems) < 2:
            # Start past the symbols already in gl. Without this the top-up
            # always returned "a0", which at level 3 is the axiom -- emitting the
            # same literal as both [axiom: a0] and [premise: a0].
            extra = _formalization_glosses(rng, 1, content, start=len(gl))
            eid = list(extra)[0]
            gl[eid] = extra[eid]
            ops.append(Operation(kind="premise", content=eid))
            oprems.append(eid)
        if len(oprems) >= 2:
            a, b = rng.sample(oprems, 2)
            ops.append(Operation(kind="prefer_premise", stronger=a, weaker=b))
    return ops, gl


_FORMALIZATION_KEY = (
    "Use these conventions, one directive per line:\n"
    "- an ordinary fact: [premise: X]\n"
    "- a fact given as certain (shown with '(certain)'): [axiom: X]\n"
    "- a rule with no modal marker is defeasible (it holds by default and can be defeated): [defeasible: A AND B => C]\n"
    "- a rule marked 'necessarily' is strict: [strict: A AND B -> C]\n"
    "- a stated preference between rules: [prefer_rule: label1 > label2]\n"
    "- a stated preference between premises: [prefer_premise: X > Y]\n"
    "- the negation of a statement: -X  (a statement listed under 'Negated statements' is the "
    "denial of an affirmative statement X and MUST be written as -X, copying X exactly; do not "
    "copy the negative sentence itself)\n"
    "Join multiple conditions with AND.")


def _negation_note(gl, ops):
    lits = set()
    for o in ops:
        if o.kind in ("premise", "axiom") and o.content.startswith("-"):
            lits.add(o.content)
        elif o.kind in ("defeasible", "strict"):
            for a in o.antecedents:
                if a.startswith("-"):
                    lits.add(a)
            if o.consequent.startswith("-"):
                lits.add(o.consequent)
    if not lits:
        return ""
    lines = ["Negated statements (each MUST be formalized as -X using its affirmative form X):"]
    for l in sorted(lits):
        a = l[1:]
        lines.append(f'  - "{gl[a]["neg"]}" is the denial of "{gl[a]["pos"]}"')
    return "\n\n" + "\n".join(lines)


def frame_formalization(rng, with_content=False, level=2):
    n = difficulty.knob('formalization', 'n_elements', level)
    if with_content:
        built = _kb_formalization_theory(rng, min(n, 9))
        if not built:
            return None
        atoms, leaves, sub = built
        ops = []
        for lit in leaves:
            kind = "axiom" if atoms.get(_ev_base(lit), {}).get("axiomatic") else "premise"
            ops.append(Operation(kind=kind, content=lit))
        di = si = 0
        for r in sub:
            if r.get("strict"):
                si += 1; nm = f"s{si}"; k = "strict"
            else:
                di += 1; nm = f"d{di}"; k = "defeasible"
            ops.append(Operation(kind=k, name=nm, antecedents=tuple(r["antecedents"]),
                                 consequent=r["consequent"]))
        if level >= 3:
            drules = [o.name for o in ops if o.kind == "defeasible"]
            oprems = [o.content for o in ops if o.kind == "premise"]
            if len(drules) >= 2:
                a, b = rng.sample(drules, 2)
                ops.append(Operation(kind="prefer_rule", stronger=a, weaker=b))
            if len(oprems) >= 2:
                a, b = rng.sample(oprems, 2)
                ops.append(Operation(kind="prefer_premise", stronger=a, weaker=b))
        gl = {a: {"pos": atoms[a]["pos"], "neg": atoms[a]["neg"]} for a in atoms}
        body = _render_content_theory(gl, ops)
        spec = ("Give the full formalization as bracketed directives, one per line. Write each "
                "statement EXACTLY as it appears in the argument - do not abbreviate or use symbols. "
                "Reference a rule by the number it is shown with, e.g. [prefer_rule: Rule 1 > Rule 2].")
        prompt = ("Translate the natural-language argument below into the formal notation.\n\n"
                  + _FORMALIZATION_KEY + "\n\nArgument:\n" + body + _negation_note(gl, ops)
                  + "\n\n" + _format_block(spec))
        # Content displays rules as "Rule N"; render the gold's rule preferences the
        # same way (index over the full displayed theory). The gold-components
        # re-parse needs DSL rule names, so relabel "Rule N" -> dN first.
        gold_dsl = _sentence_dsl(ops, gl, neg_as_minus=True, ridx=_rule_index(ops))
        gold_components = sorted(_components(parse_dsl(_relabel_rule_refs(gold_dsl)).operations))
        ref = gold_dsl
        return _entry("formalization", prompt, ref, ops, GYM_ORDERING, n_elements=len(ops),
                      mode="content", gold_components=gold_components, atoms=gl)
    ops, gl = _formalization_theory(rng, level, False)
    body = _render_content_theory(gl, ops)
    syms = list(gl)
    glossary = "\n".join(f'  "{gl[s]["pos"]}"  ->  {s}' for s in syms)
    spec = ("Give the full formalization as bracketed directives, one per line, using the "
            "glossary label for every statement.")
    prompt = ("Translate the natural-language argument below into the formal notation.\n\n"
              + _FORMALIZATION_KEY + "\n\nGlossary:\n" + glossary + "\n\nArgument:\n" + body
              + _negation_note(gl, ops) + "\n\n" + _format_block(spec))
    ref = render_dsl(ops)
    return _entry("formalization", prompt, ref, ops, GYM_ORDERING,
                  n_elements=len(ops), mode="symbolic", gold_components=sorted(_components(ops)))
