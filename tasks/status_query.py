import random
from dataclasses import replace

from aspic_gym import (
    GYM_ORDERING, ASPICVerifier, parse_dsl, JUSTIFIED, OVERRULED, UNDECIDED, sample_theory, load_kb, STATUS_LEVELS, _atoms, _gloss, _entry, _render_symbolic_theory, _render_content_theory,
)
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl
import levels as difficulty


def _bands(level):
    k = lambda n: difficulty.knob('status_query', n, level)
    return (k('prem_min'), k('prem_max'), k('rule_min'), k('rule_max'),
            k('n_axioms'), k('n_strict'), k('n_pref'))


def _status_symbolic_theory(rng, level, ordering=None, force_prefs=False):
    ordering = ordering or GYM_ORDERING
    weak = "weakest_link" in ordering
    pmn, pmx, rmn, rmx, nax, nst, npr = _bands(level)
    base = STATUS_LEVELS[level]
    easy = difficulty.gate('status_query', 'easy_no_conflict', level) and not force_prefs
    if force_prefs:
        npr = max(npr, 1)                    
        rmn = max(rmn, 2)                      
    cfg = replace(base,
                  ordering=ordering,
                  n_premises=rng.randint(pmn, pmx),
                  n_rules=rng.randint(rmn, rmx),
                  n_atoms=pmx + rmx + 3,
                  n_axioms=nax,
                  n_strict=nst,
                  p_strict=(0.0 if nst == 0 else 0.5),
                  n_pref=(max(npr, rmx) if weak else npr),
                  p_conflict=(0.0 if easy else max(base.p_conflict, 0.6 if weak else 0.5)),
                  require_conflict=(True if force_prefs
                                    else difficulty.gate('status_query', 'require_conflict', level)),
                  chain_depth=(max(0, level - 6) if level >= 9 else 0),
                  p_undercut=(0.3 if difficulty.gate('status_query', 'undercuts', level) else 0.0),
                  min_arguments=(1 if easy else 2),
                  max_tries=300)
    th = sample_theory(rng, cfg)
    return list(th.operations), th.ordering


def _status_task_block(theory_text, notation, claims_block, n):
    return (
        "You are working with a defeasible argumentation theory. " + notation + "\n\n"
        "Theory:\n" + theory_text + "\n\n"
        "Using grounded semantics (the most cautious evaluation: a claim is accepted only if every "
        "attack against it is itself defeated by accepted arguments), determine the status of each "
        f"claim below. There are {n} claim(s). Each claim's status is exactly one of:\n"
        "- justified: it has an argument that survives every attack, so it is accepted.\n"
        "- overruled: it has an argument, but a stronger surviving argument defeats it.\n"
        "- undecided: it is caught in an unresolved conflict, neither accepted nor defeated.\n"
        "- unsatisfiable: there is no argument for it at all in this theory.\n\n"
        "Claims:\n" + claims_block)


def _trim_chain(arg, K, rng):
    rules = arg.get("rules") or []
    if not rules:
        return list(arg.get("premises") or []), []
    concl = rules[-1]["consequent"]
    by_con = {}
    for r in rules:
        by_con.setdefault(r["consequent"], []).append(r)
    kept, need, seen = [], [concl], set()
    while need and len(kept) < K:
        tgt = need.pop(0)
        if tgt in seen:
            continue
        seen.add(tgt)
        cand = by_con.get(tgt)
        if not cand:
            continue
        r = rng.choice(cand)
        kept.append(r)
        if r["antecedents"]:
            need.append(r["antecedents"][0])    
    kept = kept[::-1]                           
    produced = {r["consequent"] for r in kept}
    prems = []
    for r in kept:
        for a in r["antecedents"]:
            if a not in produced and a not in prems:
                prems.append(a)
    return prems, kept


def _status_kb_theory(rng, level, ordering=None, force_prefs=False):
    kb = load_kb()
    if not kb:
        return None
    pmn, pmx, rmn, rmx, nax, nst, npr = _bands(level)
    R = rng.randint(rmn, rmx)
    cands = [r for r in kb if r.get("support") and (difficulty.gate('status_query', 'easy_single_stance', level) or r.get("disclaim"))]
    if not cands:
        return None
    rng.shuffle(cands)
    for rec in cands[:40]:
        atoms = rec["atoms"]
        sup = rng.choice(rec["support"])
        if difficulty.gate('status_query', 'undercuts', level):
            tgts = [a for a in rec.get("attacks", [])
                    if a.get("type") == "undercut" and a.get("target_stance") == "support"
                    and isinstance(a.get("target_index"), int)
                    and a["target_index"] < len(rec["support"])]
            if tgts:
                sup = rec["support"][rng.choice(tgts)["target_index"]]
        if difficulty.gate('status_query', 'easy_single_stance', level):
            ks, dis_want, kd = R, 0, 0
        else:
            dis_want = 2 if (npr >= 2 and len(rec["disclaim"]) >= 2) else 1
            kd = max(1, R // 3)
            ks = max(1, R - kd)
        sp, sr = _trim_chain(sup, ks, rng)
        if not sr:
            continue
        dis_chains = []
        if dis_want:
            dchoices = list(rec["disclaim"]); rng.shuffle(dchoices)
            for d in dchoices[:dis_want]:
                dp, dr = _trim_chain(d, max(1, kd // dis_want), rng)
                if dr:
                    dis_chains.append((dp, dr))
            if not dis_chains:
                continue
        facts = list(dict.fromkeys(sp + [p for dp, _ in dis_chains for p in dp]))
        ax_atoms = set()
        if nax:
            ax_cand = [p for p in facts if atoms.get(p, {}).get("axiomatic")] or facts
            ax_atoms = set(ax_cand[:nax])
        lines = [f"[{'axiom' if p in ax_atoms else 'premise'}: {p}]" for p in facts]
        rule_rs = list(sr) + [r for _, dr in dis_chains for r in dr]
        di, strict_left, sup_final, dis_finals = 0, nst, None, []
        kbname_to_local = {}
        uc_targets = set()
        if difficulty.gate('status_query', 'undercuts', level):
            uc_targets = {a.get("target_point") for a in rec.get("attacks", [])
                          if a.get("type") == "undercut"}
        for r in rule_rs:
            di += 1; nm = f"d{di}"
            is_final = r["consequent"] in ("c0", "-c0")
            strict = strict_left > 0 and not is_final and r.get("name") not in uc_targets
            if strict:
                strict_left -= 1
            arrow = "->" if strict else "=>"
            kw = "strict" if strict else "defeasible"
            lines.append(f"[{kw} {nm}: {' AND '.join(r['antecedents'])} {arrow} {r['consequent']}]")
            if not strict and r.get("name"):
                kbname_to_local[r["name"]] = nm
            if r["consequent"] == "c0":
                sup_final = nm
            elif r["consequent"] == "-c0":
                dis_finals.append(nm)
        if difficulty.gate('status_query', 'undercuts', level) and kbname_to_local:
            for atk in rec.get("attacks", []):
                if atk.get("type") != "undercut":
                    continue
                local = kbname_to_local.get(atk.get("target_point"))
                if not local:
                    continue
                for pr in atk.get("premises", []):
                    if pr not in facts:
                        facts.append(pr)
                        lines.append(f"[premise: {pr}]")
                for r2 in atk.get("rules", []):
                    di += 1
                    lines.append(f"[defeasible d{di}: "
                                 f"{' AND '.join(r2['antecedents'])} => -{local}]")
                break
        ops = list(parse_dsl("\n".join(lines)).operations)
        if (force_prefs or (npr and difficulty.gate('status_query', 'kb_prefs', level))) \
                and sup_final and dis_finals:
            pls = []
            weak = "weakest_link" in (ordering or "")
            ordinary = {o.content for o in ops
                        if getattr(o, "kind", "") == "premise"}
            sup_prems = [p for p in dict.fromkeys(sp) if p in ordinary]
            dis_prems = [p for p in dict.fromkeys(pp for dp, _ in dis_chains for pp in dp)
                         if p in ordinary]
            eff_npr = max(npr, 1) if force_prefs else npr
            # The coin flip is re-rolled per disclaim-final, so two passes with
            # opposite outcomes would otherwise emit both p > q and q > p over
            # the same pair. That is legal (the ordering is a preorder, so it
            # reads as a tie) but it is undeclared, ungated, and unmentioned in
            # the prompt -- an accident rather than a designed feature.
            seen_r, seen_p = set(), set()
            for df in dis_finals[:eff_npr]:
                sup_wins = rng.random() < 0.5
                a, b = (sup_final, df) if sup_wins else (df, sup_final)
                if (a, b) not in seen_r and (b, a) not in seen_r:
                    seen_r.add((a, b))
                    pls.append(f"[prefer_rule: {a} > {b}]")
                if weak:
                    win_p, lose_p = (sup_prems, dis_prems) if sup_wins else (dis_prems, sup_prems)
                    for wp in win_p:
                        for lp in lose_p:
                            if (wp != lp and (wp, lp) not in seen_p
                                    and (lp, wp) not in seen_p):
                                seen_p.add((wp, lp))
                                pls.append(f"[prefer_premise: {wp} > {lp}]")
            if pls:
                ops += list(parse_dsl("\n".join(pls)).operations)
        if "c0" not in _atoms(ops):
            continue
        return rec, atoms, ops
    return None


def frame_status_query(rng, with_content=False, level=2, ordering=None):
    n = difficulty.knob('status_query', 'n_claims', level)   
    if with_content:
        ordering = ordering or GYM_ORDERING
        built = _status_kb_theory(rng, level, ordering)
        if not built:
            return None
        rec, atoms, ops = built
        v = ASPICVerifier.from_operations(ops, ordering=ordering)
        syms = _atoms(ops)                     
        all_lits = [s for s in syms] + ["-" + s for s in syms]
        gloss_fn = lambda l: _gloss(atoms, l)
        theory_text = _render_content_theory(atoms, ops)
        notation = _CONTENT_NOTATION
    else:
        ops, ordering = _status_symbolic_theory(rng, level, ordering or GYM_ORDERING)
        v = ASPICVerifier.from_operations(ops, ordering=ordering)
        syms = _atoms(ops)
        all_lits = [a for a in syms] + ["-" + a for a in syms]
        gloss_fn = lambda l: l
        theory_text = _render_symbolic_theory(ops)
        notation = _sym_notation(level)
    sm = v.status_map()
    _base = lambda l: l[1:] if l.startswith("-") else l
    fact_bases = {o.content for o in ops if o.kind in ("premise", "axiom")}
    derived_bases = {o.consequent for o in ops if o.kind in ("defeasible", "strict")}
    def _role(l):
        b = _base(l)
        if b in derived_bases:
            return "derived"
        if b in fact_bases:
            return "fact"
        return "absent"
    if difficulty.gate('status_query', 'kb_perturb', level):
        all_lits = [l for l in all_lits
                    if _base(l) in derived_bases and _base(l) not in fact_bases]
    J = [l for l in all_lits if sm.get(l) == JUSTIFIED]
    O = [l for l in all_lits if sm.get(l) == OVERRULED]
    U = [l for l in all_lits if sm.get(l) == UNDECIDED]
    unsat = [l for l in all_lits if l not in sm]
    _cons = {o.consequent: o for o in ops if o.kind in ("defeasible", "strict")}
    def _depth(l, seen=None):
        seen = seen or set()
        r = _cons.get(l)
        if not r or l in seen:
            return 0
        return 1 + max([_depth(a, seen | {l}) for a in (r.antecedents or ())] + [0])
    depth_bias = level >= 5
    for b in (J, O, U, unsat):
        rng.shuffle(b)
        if depth_bias:                       
            b.sort(key=lambda l: _depth(l), reverse=True)
    buckets = [("JUSTIFIED", J), ("OVERRULED", O), ("UNDECIDED", U), ("UNSATISFIABLE", unsat)]
    rng.shuffle(buckets)
    if n == 1:
        nonempty = [(nm, b) for nm, b in buckets if b]
        if not nonempty:
            return None
        nm, b = nonempty[rng.randrange(len(nonempty))]
        pool = [(nm, b[0] if depth_bias else b[rng.randrange(len(b))])]
    else:
        nonempty = [list(b) for _nm, b in buckets if b]
        names = [nm for nm, b in buckets if b]
        pool = []
        i = 0
        while any(nonempty):
            k = i % len(nonempty)
            if nonempty[k]:
                pool.append((names[k], nonempty[k].pop(0) if depth_bias else nonempty[k].pop()))
            if not nonempty[k]:
                nonempty.pop(k); names.pop(k)
                i = k
            else:
                i += 1
    claims, gstat, used_keys = [], [], set()
    def _pick(prefer_new):
        for nm, l in pool:
            if l in claims:
                continue
            if prefer_new and (nm, _role(l)) in used_keys:
                continue
            return nm, l
        return None
    if n > 1:
        n = max(2, min(n, len({l for _nm, l in pool})))
    while len(claims) < n:
        got = _pick(True) or _pick(False)
        if not got:
            break
        nm, l = got
        claims.append(l); gstat.append(nm); used_keys.add((nm, _role(l)))
    if len(claims) < n:
        return None
    if n >= 2 and len(set(gstat)) < 2:
        return None                                
    if n >= 3 and max(gstat.count(s) for s in set(gstat)) > (n + 2) // 3 + 1:
        return None
    if n >= 3 and gstat.count("JUSTIFIED") > (n + 1) // 2:
        return None                                
    order = list(range(len(claims))); rng.shuffle(order)
    claims = [claims[i] for i in order]; gstat = [gstat[i] for i in order]
    claims_block = "\n".join(f"  {i + 1}. {gloss_fn(c)}" for i, c in enumerate(claims))
    task_block = _status_task_block(theory_text, notation, claims_block, n)
    fmt = _format_block(
        "In the answer, write one line for each claim in the form `number: status` (for example "
        "`1: justified`). Use only the words justified, overruled, undecided, or unsatisfiable.")
    order_decl = _ordering_decl(ordering)
    prompt = _intro(level) + "\n\n" + order_decl + task_block + "\n\n" + fmt
    reference = "[answer]\n" + "\n".join(f"{i + 1}: {s.lower()}" for i, s in enumerate(gstat)) + "\n[/answer]"
    return _entry("status_query", prompt, reference, ops, ordering,
                  queries=claims, gold_statuses=gstat, n_claims=n,
                  mode=("content" if with_content else "symbolic"),
                  claim_display=[gloss_fn(c) for c in claims])
