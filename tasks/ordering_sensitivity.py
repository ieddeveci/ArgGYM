import random

from aspic_gym import (
    GYM_ORDERING, ASPICVerifier, JUSTIFIED, OVERRULED, UNDECIDED, sample_theory, STATUS_LEVELS,
    _entry, _render_symbolic_theory, _render_content_theory, _gloss, load_kb,
)
from tasks.status_query import _status_symbolic_theory, _status_kb_theory
from prompting import _format_block, _intro, _sym_notation, _CONTENT_NOTATION
import levels as difficulty

_LAST = "last_link_elitist"
_WEAK = "weakest_link_elitist"


def _status_name(v, lit):
    sm = v.status_map()
    s = sm.get(lit)
    if s is None:
        return "unsatisfiable"
    return {JUSTIFIED: "justified", OVERRULED: "overruled",
            UNDECIDED: "undecided"}.get(s, str(s).lower())


def _divergent_claim(ops):
    vL = ASPICVerifier.from_operations(ops, ordering=_LAST)
    vW = ASPICVerifier.from_operations(ops, ordering=_WEAK)
    mL, mW = vL.status_map(), vW.status_map()
    cands = []
    for k in set(mL) | set(mW):
        if k.startswith("-") or k.lstrip("-").startswith("d") or k.lstrip("-").startswith("s"):
            continue
        sl, sw = _status_name(vL, k), _status_name(vW, k)
        if sl != sw:
            cands.append((k, sl, sw))
    return cands


def _build_divergence_theory(rng, level, want_pair):
    from aspic_gym import parse_dsl
    (last_s, weak_s) = want_pair
    dA = min(2 + level // 4, 5)
    dB = min(2 + level // 5, 5)
    tag = f"o{rng.randint(0, 9_000_000)}_"
    ops = []

    def chain(prefix, concl, depth, src):
        ops.append(f"[premise: {tag}{src}]")
        prev, rules = f"{tag}{src}", []
        for i in range(depth):
            r = f"{prefix}{i+1}"
            nxt = concl if i == depth - 1 else f"{tag}{prefix}n{i}"
            ops.append(f"[defeasible {r}: {prev} => {nxt}]")
            rules.append(r); prev = nxt
        return rules

    x = f"{tag}x"
    ra = chain("a", x, dA, "p")
    rb = chain("b", "-" + x, dB, "q")

    if weak_s == "undecided":
        if last_s == "justified":
            ops.append(f"[prefer_rule: {ra[-1]} > {rb[-1]}]")
        else:
            ops.append(f"[prefer_rule: {rb[-1]} > {ra[-1]}]")
    else:
        if last_s == "justified":            
            ops.append(f"[prefer_rule: {ra[-1]} > {rb[-1]}]")
            for wr in rb:
                ops.append(f"[prefer_rule: {wr} > {ra[0]}]")
            ops.append(f"[prefer_premise: {tag}q > {tag}p]")
        else:                                
            ops.append(f"[prefer_rule: {rb[-1]} > {ra[-1]}]")
            for wr in ra:
                ops.append(f"[prefer_rule: {wr} > {rb[0]}]")
            ops.append(f"[prefer_premise: {tag}p > {tag}q]")

    for _ in range(max(0, level // 3 - 1)):
        d1, z = f"{tag}d{rng.randint(0,99999)}", f"{tag}z{rng.randint(0,99999)}"
        ops.append(f"[premise: {d1}]")
        ops.append(f"[defeasible k{rng.randint(0,99999)}: {d1} => {z}]")

    parsed = list(parse_dsl("\n".join(ops)).operations)
    vL = ASPICVerifier.from_operations(parsed, ordering=_LAST)
    vW = ASPICVerifier.from_operations(parsed, ordering=_WEAK)
    if _status_name(vL, x) == last_s and _status_name(vW, x) == weak_s and vL.is_consistent():
        return parsed, x
    return None


def _order_legend():
    return (
        "This theory can be evaluated under two different orderings, which decide how a "
        "preference makes one argument beat another:\n"
        "- LAST-LINK: an argument's strength is decided by its FINAL defeasible rule, so a "
        "preference on that final rule is enough to make the argument prevail.\n"
        "- WEAKEST-LINK: an argument is only as strong as its WEAKEST defeasible rule, so it "
        "prevails only if it outranks the opponent across its whole chain (rules and premises).")


def _build_content_divergence(rng, level, want_pair, kb_atoms):
    from aspic_gym import parse_dsl
    last_s, weak_s = want_pair
    ids = [k for k in kb_atoms.keys() if kb_atoms[k].get("pos")]
    budget = len(ids)
    if budget < 6:                    
        return None
    dA = min(2 + level // 4, 4)
    dB = min(2 + level // 5, 4)
    n_decoy = max(0, level // 3 - 1)
    while 2 + (dA - 1) + (dB - 1) + 1 + 2 * n_decoy > budget:
        if n_decoy > 0:
            n_decoy -= 1
        elif dB > 2:
            dB -= 1
        elif dA > 2:
            dA -= 1
        else:
            break
    if 2 + (dA - 1) + (dB - 1) + 1 > budget:
        return None
    rng.shuffle(ids)
    it = iter(ids)
    nxt = lambda: next(it)
    ops = []
    atoms_used = {}

    def use(aid):
        atoms_used[aid] = kb_atoms[aid]
        return aid

    def chain(prefix, concl, depth, src):
        ops.append(f"[premise: {use(src)}]")
        prev, rules = src, []
        for i in range(depth):
            r = f"{prefix}{i+1}"
            nxt_atom = concl if i == depth - 1 else use(nxt())
            ops.append(f"[defeasible {r}: {prev} => {nxt_atom}]")
            rules.append(r); prev = nxt_atom
        return rules

    try:
        p, q, x = nxt(), nxt(), use(nxt())
        ra = chain("a", x, dA, p)
        rb = chain("b", "-" + x, dB, q)
    except StopIteration:
        return None

    if weak_s == "undecided":
        if last_s == "justified":
            ops.append(f"[prefer_rule: {ra[-1]} > {rb[-1]}]")
        else:
            ops.append(f"[prefer_rule: {rb[-1]} > {ra[-1]}]")
    else:
        if last_s == "justified":
            ops.append(f"[prefer_rule: {ra[-1]} > {rb[-1]}]")
            for wr in rb:
                ops.append(f"[prefer_rule: {wr} > {ra[0]}]")
            ops.append(f"[prefer_premise: {q} > {p}]")
        else:
            ops.append(f"[prefer_rule: {rb[-1]} > {ra[-1]}]")
            for wr in ra:
                ops.append(f"[prefer_rule: {wr} > {rb[0]}]")
            ops.append(f"[prefer_premise: {p} > {q}]")

    for _ in range(n_decoy):                    
        try:
            d1, z = use(nxt()), use(nxt())
        except StopIteration:
            break
        ops.append(f"[premise: {d1}]")
        ops.append(f"[defeasible k{rng.randint(0,99999)}: {d1} => {z}]")

    try:
        parsed = list(parse_dsl("\n".join(ops)).operations)
        vL = ASPICVerifier.from_operations(parsed, ordering=_LAST)
        vW = ASPICVerifier.from_operations(parsed, ordering=_WEAK)
        if _status_name(vL, x) == last_s and _status_name(vW, x) == weak_s and vL.is_consistent():
            return parsed, x, atoms_used
    except Exception:
        return None
    return None


def _inject_content_divergence(ops, want_pair, rng):
    from aspic_gym import parse_dsl, Operation
    last_s, weak_s = want_pair
    if weak_s == "undecided":
        return None                      
    c0 = "c0"
    rules = [o for o in ops if o.kind == "defeasible" and o.name]
    cons = {}
    for o in ops:
        if o.kind in ("defeasible", "strict"):
            cons.setdefault(o.consequent, o)
    prem = {o.content for o in ops if o.kind == "premise"}

    def chain_rules_prems(head):
        seen, stack, rs, ps = set(), [head], [], []
        while stack:
            lit = stack.pop()
            if lit in seen:
                continue
            seen.add(lit)
            if lit in prem:
                ps.append(lit)
            sub = cons.get(lit)
            if sub is not None:
                if sub.kind == "defeasible" and sub.name:
                    rs.append(sub.name)
                stack.extend(sub.antecedents or ())
        return rs, ps

    aR, aP = chain_rules_prems(c0)
    bR, bP = chain_rules_prems("-c0")
    a_last = next((o.name for o in ops if o.kind == "defeasible" and o.consequent == c0), None)
    b_last = next((o.name for o in ops if o.kind == "defeasible" and o.consequent == "-c0"), None)
    if not a_last or not b_last or len(aR) < 2 or not aP or not bP:
        return None
    base = [o for o in ops if o.kind not in ("prefer_rule", "prefer_premise")]
    pls = []
    if last_s == "justified":           
        pls.append(f"[prefer_rule: {a_last} > {b_last}]")
        a_early = next(r for r in aR if r != a_last)
        for wr in bR:
            pls.append(f"[prefer_rule: {wr} > {a_early}]")
        for wp in bP:
            for lp in aP:
                if wp != lp:
                    pls.append(f"[prefer_premise: {wp} > {lp}]")
    else:                                  
        pls.append(f"[prefer_rule: {b_last} > {a_last}]")
        b_early = next((r for r in bR if r != b_last), None)
        if not b_early:
            return None
        for wr in aR:
            pls.append(f"[prefer_rule: {wr} > {b_early}]")
        for wp in aP:
            for lp in bP:
                if wp != lp:
                    pls.append(f"[prefer_premise: {wp} > {lp}]")
    try:
        new_ops = base + list(parse_dsl("\n".join(pls)).operations)
        vL = ASPICVerifier.from_operations(new_ops, ordering=_LAST)
        vW = ASPICVerifier.from_operations(new_ops, ordering=_WEAK)
        if _status_name(vL, c0) == last_s and _status_name(vW, c0) == weak_s and vL.is_consistent():
            return new_ops, c0
    except Exception:
        return None
    return None


_PAIR_TYPES = [("justified", "overruled"), ("overruled", "justified"),
               ("justified", "undecided"), ("overruled", "undecided")]


def frame_ordering_sensitivity(rng, with_content=False, level=2):
    if not with_content:
        want = _PAIR_TYPES[rng.randrange(len(_PAIR_TYPES))]
        built = None
        for _ in range(40):
            built = _build_divergence_theory(rng, level, want)
            if built:
                break
        if not built:                     
            for alt in _PAIR_TYPES:
                built = _build_divergence_theory(rng, level, alt)
                if built:
                    break
        if not built:
            return None
        ops, claim = built
        vL = ASPICVerifier.from_operations(ops, ordering=_LAST)
        vW = ASPICVerifier.from_operations(ops, ordering=_WEAK)
        last_s, weak_s = _status_name(vL, claim), _status_name(vW, claim)
        atoms = None
        gloss = lambda l: l
        theory_text = _render_symbolic_theory(ops)
        notation = _sym_notation(level)
        mode = "symbolic"
    else:
        want = _PAIR_TYPES[rng.randrange(len(_PAIR_TYPES))]
        kb = load_kb(required=False) or []
        recs = [r for r in kb if r.get("atoms")]
        built = None
        if recs:
            for _ in range(40):
                rec = recs[rng.randrange(len(recs))]
                r = _build_content_divergence(rng, level, want, rec["atoms"])
                if not r:
                    continue
                ops, claim, atoms = r
                vL = ASPICVerifier.from_operations(ops, ordering=_LAST)
                vW = ASPICVerifier.from_operations(ops, ordering=_WEAK)
                last_s, weak_s = _status_name(vL, claim), _status_name(vW, claim)
                built = True
                break
        if not built:
            return None
        gloss = lambda l: _gloss(atoms, l)
        theory_text = _render_content_theory(atoms, ops)
        notation = _CONTENT_NOTATION
        mode = "content"

    spec = ("Give the status of the claim under EACH ordering, on two lines, exactly:\n"
            "last-link: <status>\nweakest-link: <status>\n"
            "where <status> is one of justified, overruled, undecided, unsatisfiable.")
    task_block = (
        "You are analysing a defeasible argumentation theory. " + notation + "\n\n"
        "Theory:\n" + theory_text + "\n\n" + _order_legend() + "\n\n"
        f"For the claim \"{gloss(claim)}\" determine its grounded status under last-link "
        "ordering and, separately, under weakest-link ordering.")
    prompt = _intro(level) + "\n\n" + task_block + "\n\n" + _format_block(spec)
    ref = ("[answer]\n" + f"last-link: {last_s}\n" + f"weakest-link: {weak_s}\n" + "[/answer]")
    meta = dict(claim=claim, last_status=last_s, weak_status=weak_s, mode=mode)
    if with_content:
        meta["atoms"] = atoms
    return _entry("ordering_sensitivity", prompt, ref, ops, _LAST, **meta)
