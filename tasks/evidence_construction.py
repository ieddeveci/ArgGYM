
from aspic_gym import (
    GYM_ORDERING, Operation, parse_dsl, ops_to_dicts, ASPICVerifier, JUSTIFIED, load_kb, _ev_base, _ev_leaning, _sentence_dsl, _entry,
)
from prompting import _format_block, _intro
import levels as difficulty

def EVIDENCE_LEVELS_AT(level):
    k = lambda n: difficulty.knob('evidence_construction', n, level)
    return {'n_distract': k('n_distract'), 'n_oppose': k('n_oppose'), 'min_prem': k('min_prem')}


def _ev_arg_literals(arg):
    lits = list(arg["premises"]) + [r["consequent"] for r in arg["rules"]]
    lits += [x for r in arg["rules"] for x in r["antecedents"]]
    return lits


def _ev_witness_ok(atoms, arg, stance):
    bad = "con" if stance == "support" else "pro"
    return all(_ev_leaning(atoms, L) != bad for L in arg["premises"])


def _ev_render_dsl(arg):
    lines = [f"[premise: {p}]" for p in arg["premises"]]
    for r in arg["rules"]:
        lines.append(f"[defeasible: {' AND '.join(r['antecedents'])} => {r['consequent']}]")
    return "\n".join(lines)


def _ev_arg_to_ops(atoms, arg):
    ops = []
    for lit in arg["premises"]:
        ops.append(Operation(kind="premise", content=lit))
    for lit in arg.get("axioms", []):
        kk = "axiom" if atoms.get(lit.lstrip("-"), {}).get("axiomatic") else "premise"
        ops.append(Operation(kind=kk, content=lit))
    for r in arg["rules"]:
        kk = "strict" if r.get("strict") else "defeasible"
        ops.append(Operation(kind=kk, antecedents=tuple(r["antecedents"]), consequent=r["consequent"]))
    return ops


def frame_evidence_construction(rng, with_content=False, level=2):
    kb = load_kb()
    if not kb:
        return None
    cfg = EVIDENCE_LEVELS_AT(level)
    order = list(range(len(kb)))
    rng.shuffle(order)
    for ri in order:
        rec = kb[ri]
        atoms = rec["atoms"]
        stance = rng.choice(["support", "disclaim"])
        target = "c0" if stance == "support" else "-c0"
        opp_stance = "disclaim" if stance == "support" else "support"
        cands = [a for a in rec.get(stance, [])
                 if _ev_witness_ok(atoms, a, stance) and not a.get("axioms")
                 and len(a["premises"]) >= cfg["min_prem"]]
        if not cands:
            continue
        witness = rng.choice(cands)
        w_atoms = {_ev_base(L) for L in _ev_arg_literals(witness) if _ev_base(L) != "c0"}
        others = [i for i in atoms if i != "c0" and i not in w_atoms]
        rng.shuffle(others)
        pool_ids = sorted(w_atoms) + others[:cfg["n_distract"]]   
        rng.shuffle(pool_ids)

        base_ops, seeded_rule_names = [], []

        wit = _ev_render_dsl(witness).replace("\n", "")
        for i, rn in enumerate(seeded_rule_names):
            wit += f"[premise: uw{i}][defeasible: uw{i} => -{rn}]"
        try:
            cnt = {"d": sum(1 for o in base_ops if o.kind == "defeasible"),
                   "s": sum(1 for o in base_ops if o.kind == "strict")}
            vchk = ASPICVerifier.from_operations(
                base_ops + list(parse_dsl(wit, counters=cnt).operations), ordering=GYM_ORDERING)
            if not (vchk.status(target) == JUSTIFIED and vchk.is_consistent()):
                continue
        except Exception:
            continue

        producers = {}
        for a in rec.get("support", []) + rec.get("disclaim", []):
            for r in a["rules"]:
                producers.setdefault(r["consequent"], []).append(list(r["antecedents"]))

        target_nl = atoms["c0"]["pos"] if stance == "support" else atoms["c0"]["neg"]
        if with_content:
            gl = {a: {"pos": atoms[a]["pos"], "neg": atoms[a]["neg"]} for a in atoms}
            pool_lines = "\n".join(f'  - "{atoms[i]["pos"]}"' for i in pool_ids)
            how = ('Build your argument from the statements above (use a statement as itself, or '
                   'negated by writing -"the statement" or "not the statement"). Write premises as '
                   '[premise: <statement>] and rules as [defeasible: <statement> AND ... => '
                   f'<statement>], finishing at the conclusion "{target_nl}". Every premise you '
                   'assert must be one of the listed statements (or its negation); an argument '
                   'standing on unlisted premises scores nothing. Your argument must stand '
                   f'on at least {cfg["min_prem"]} distinct listed statement(s), and every '
                   'premise you rely on must genuinely favour the conclusion you are arguing for.')
            theory_block = ""
            ref_sol = _sentence_dsl(_ev_arg_to_ops(atoms, witness), gl)
            mode = "content"
        else:
            pool_lines = "\n".join(f'  {i} = "{atoms[i]["pos"]}"' for i in pool_ids)
            how = (f"Use the statements as starting points (each as itself, e.g. {pool_ids[0]}, or "
                   f"negated, e.g. -{pool_ids[0]}). Write premises as [premise: {pool_ids[0]}] and "
                   f"rules as [defeasible: {pool_ids[0]} AND ... => {target}], finishing at "
                   f"{target}. Every premise you assert must come from the listed statements (as "
                   "itself or negated); an argument standing on unlisted premises scores nothing. Your "
                   f"argument must stand on at least {cfg['min_prem']} distinct listed statements, and every "
                   "premise you rely on must genuinely favour the conclusion you are arguing for.")
            theory_block = ""
            ref_sol = wit.replace("][", "]\n[")
            mode = "symbolic"

        task = (f'Claim: "{atoms["c0"]["pos"]}"\n\nConstruct an argument that makes the following '
                f'conclusion hold under grounded semantics: "{target_nl}".\n\nStatements:\n'
                + pool_lines + theory_block + "\n\n" + how)
        spec = ("Lay out your reasoning in the reasoning block. The answer block must contain only "
                "your argument's directives, one per line.")
        prompt = _intro(level) + "\n\n" + task + "\n\n" + _format_block(spec)
        ref = "[answer]\n" + ref_sol + "\n[/answer]"
        return _entry("evidence_construction", prompt, ref, [], GYM_ORDERING,
                      claim=rec["claim"], stance=stance, target=target, atoms=atoms, mode=mode,
                      min_premises=cfg["min_prem"],
                      pool=pool_ids, base_ops=ops_to_dicts(base_ops),
                      seeded_rules=seeded_rule_names, n_premises=len(witness["premises"]),
                      producers=producers)
    return None
