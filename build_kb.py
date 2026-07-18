#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    from aspic_api import ASPICVerifier
    from aspic_engine import contrary
except Exception as e:                                   
    sys.stderr.write(f"Could not import the ASPIC+ framework ({e}). "
                     "Run from the directory with aspic_*.py.\n")
    raise

from generate_argumentations import _ollama_chat, _parse_json, _atomic_problems, _san

JUSTIFIED = "JUSTIFIED"

def normalize_claim(s):

    import re as _re
    s = _re.sub(r"\s+", " ", (s or "")).strip()
    if s.endswith("."):
        s = s[:-1]
    return s.casefold()


def dedupe_claims(claims, verbose=True):
    seen, out = set(), []
    for c in claims:
        k = normalize_claim(c)
        if k in seen:
            if verbose:
                print(f"    [dedupe] skipping duplicate claim: {c[:60]!r}")
            continue
        seen.add(k)
        out.append(c)
    return out

PROFILES = {
    "max": dict(n_props=20, n_each=7, tier_max=4,
                attacks=dict(rebut=2, undermine=2, undercut=2), derived=True),
}

INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_negation": {"type": "string"},
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pos": {"type": "string"},
                    "neg": {"type": "string"},
                    "polarity": {"type": "string", "enum": ["pro", "con", "neutral"]},
                    "tier": {"type": "integer", "minimum": 0, "maximum": 4},
                    "axiomatic": {"type": "boolean"},
                },
                "required": ["pos", "neg", "polarity", "tier"],
            },
        },
    },
    "required": ["claim_negation", "propositions"],
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "polarity_fixes": {"type": "object", "additionalProperties": {"type": "string"}},
        "tier_fixes": {"type": "object", "additionalProperties": {"type": "integer"}},
        "axiom_fixes": {"type": "object", "additionalProperties": {"type": "boolean"}},
        "nonatomic": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["merges", "polarity_fixes", "nonatomic"],
}

ARG_SCHEMA = {
    "type": "object",
    "properties": {
        "premises": {"type": "array", "items": {"type": "string"}},
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "antecedents": {"type": "array", "items": {"type": "string"}},
                    "consequent": {"type": "string"},
                    "strict": {"type": "boolean"},
                },
                "required": ["antecedents", "consequent"],
            },
        },
        "conclusion": {"type": "string"},
    },
    "required": ["premises", "rules", "conclusion"],
}

def _aslit(x):
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, dict):
        for k in ("literal", "id", "value", "name", "premise", "antecedent",
                  "consequent", "atom", "proposition", "prop", "ref", "text"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def base(lit):
    if not isinstance(lit, str):
        lit = _aslit(lit) or ""
    return lit[1:] if lit.startswith("-") else lit


def tier_of(atoms, lit):
    return atoms.get(base(lit), {}).get("tier", 0)


def rules_climb(atoms, rules, strict=False):
    for r in rules:
        ct = tier_of(atoms, r["consequent"])
        mx = max((tier_of(atoms, a) for a in r["antecedents"]), default=0)
        if (ct <= mx) if strict else (ct < mx):
            return False
    return True


def gloss(atoms, lit):
    a = base(lit)
    if a not in atoms:
        return lit
    return atoms[a]["neg"] if lit.startswith("-") else atoms[a]["pos"]


def _clean_rules(atoms, rules):
    out = []
    for r in rules:
        ants = list(r.get("antecedents") or [])
        con = r.get("consequent")
        if not ants or not con:
            continue
        if base(con) in atoms and all(base(x) in atoms for x in ants):
            out.append({"antecedents": ants, "consequent": con,
                        "strict": bool(r.get("strict", False)), "name": r.get("name")})
    return out


def _leaves(atoms, premises, rules):
    rules = _clean_rules(atoms, rules)
    prem = {p for p in premises if base(p) in atoms}
    produced = {r["consequent"] for r in rules}
    used = {x for r in rules for x in r["antecedents"]}
    prem |= {x for x in used if x not in produced}
    return prem


def build_dsl(atoms, premises, rules, conclusion):
    rules = _clean_rules(atoms, rules)
    prem = _leaves(atoms, premises, rules)
    lines = []
    for p in sorted(prem):
        kind = "axiom" if atoms.get(base(p), {}).get("axiomatic") else "premise"
        lines.append(f"[{kind}: {p}]")
    for r in rules:
        kw = "strict" if r["strict"] else "defeasible"
        arrow = "->" if r["strict"] else "=>"
        label = f' {r["name"]}' if r.get("name") else ""
        lines.append(f"[{kw}{label}: {' AND '.join(r['antecedents'])} {arrow} {r['consequent']}]")
    return "\n".join(lines)


def render_readable(atoms, premises, rules, conclusion):
    rules = _clean_rules(atoms, rules)
    leaves = _leaves(atoms, premises, rules)
    out = []
    if leaves:
        out.append("Given facts:")
        for pr in sorted(leaves):
            certain = " (certain)" if atoms.get(base(pr), {}).get("axiomatic") else ""
            out.append(f"  - {gloss(atoms, pr)}{certain}")
    if rules:
        out.append("Rules:")
        for i, r in enumerate(rules, 1):
            ants = " and ".join(gloss(atoms, a) for a in r["antecedents"])
            link = "then necessarily" if r["strict"] else "then"
            out.append(f"  Rule {r.get('name') or i}: if {ants}, {link} "
                       f"{gloss(atoms, r['consequent'])}.")
    out.append("THEREFORE: " + gloss(atoms, conclusion))
    return "\n".join(out)


def verify(dsl, conclusion, ordering="last_link_elitist"):
    try:
        v = ASPICVerifier.from_dsl(dsl, ordering=ordering)
    except Exception:
        return False, None
    if not v.is_consistent() or not v.check_invariants().ok:
        return False, None
    st = v.status(conclusion)
    return st == JUSTIFIED, st


def arg_signature(premises, rules, conclusion):
    shape = frozenset((tuple(sorted(r["antecedents"])), r["consequent"]) for r in rules)
    return hash((frozenset(premises), shape, conclusion))

INV_SYSTEM = (
    "You are an argumentation knowledge engineer. Given a CLAIM, produce a compact inventory of "
    "ATOMIC propositions relevant to deciding it. For each proposition give `pos` (an affirmative "
    "atomic sentence: ONE simple present-tense fact, no if/then, no because/therefore/implies, do "
    "not join two facts with and/or, and no vague intensifiers such as entirely, fully, only, "
    "solely, genuine, always, never), `neg` (the natural negation of pos, also atomic), and "
    "`polarity` of the POS form judged IN ISOLATION relative to the claim: 'pro' if pos by itself "
    "makes the claim more likely true, 'con' if it makes it less likely, 'neutral' otherwise. "
    "Also give `tier`, the evidential level: 0 = a foundational, self-evident starting fact (a "
    "direct observation, a definition, or something granted without argument); 1 = an intermediate "
    "proposition that only becomes reasonable once some tier-0 facts are established; 2 = a "
    "substantive, near-claim conclusion that depends on lower tiers being settled first. Aim for "
    "several tier-0 building blocks and fewer tier-2 propositions. "
    "Also give `axiomatic` (true/false): set it TRUE only when the proposition is incontrovertible "
    "- true purely by definition, logic, or mathematics, so that denying it is incoherent (for "
    "example 'a triangle has three sides', 'a bachelor is an unmarried man', 'every number equals "
    "itself', 'either it is raining or it is not raining'). An axiomatic proposition is NOT an "
    "empirical fact that could turn out false, NOT a strong opinion, and NOT a merely well-"
    "established or very likely fact - it must hold by meaning alone. Set axiomatic=false for "
    "everything else; most propositions are NOT axiomatic, so use true sparingly, and only at "
    "tier 0. "
    "Include facts that supporters AND opponents would invoke, plus neutral background facts. Also "
    "give `claim_negation`: the natural negation of the claim. Return JSON only."
)


def build_inventory(claim, model, host, temp, api_key, n_props=14, tier_max=2):
    tier_note = (f" Use evidential tiers 0..{tier_max}: 0 = foundational, self-evident starting "
                 f"fact; each higher tier is a proposition that only becomes reasonable once "
                 f"lower-tier ones are established; {tier_max} = a substantive, near-claim "
                 f"conclusion. Provide several tier-0 building blocks and progressively fewer "
                 f"at each higher tier.")
    user = (f'CLAIM: "{claim}"\nProduce about {n_props} atomic propositions (a mix of pro, con, '
            "and neutral) plus claim_negation." + tier_note)
    raw = _ollama_chat(model, INV_SYSTEM, user, host, temp, api_key, schema=INVENTORY_SCHEMA)
    data = _parse_json(raw)
    if not data or not isinstance(data, dict):
        return None
    atoms = {"c0": {"pos": _san(claim), "neg": _san(data.get("claim_negation", "not " + claim)),
                    "polarity": "pro", "tier": tier_max + 1, "axiomatic": False}}
    i = 0
    for p in data.get("propositions", []):
        pos, neg = _san(p.get("pos", "")), _san(p.get("neg", ""))
        if not pos or not neg:
            continue
        if _atomic_problems(pos) or _atomic_problems(neg):
            continue                                      
        i += 1
        tier = p.get("tier", 0)
        tier = tier if isinstance(tier, int) and 0 <= tier <= tier_max else 0
        ax = bool(p.get("axiomatic", False)) and tier == 0 
        atoms[f"q{i}"] = {"pos": pos, "neg": neg,
                          "polarity": p.get("polarity", "neutral"), "tier": tier,
                          "axiomatic": ax}
    return atoms

CRITIC_SYSTEM = (
    "You review a proposition inventory for a claim. Return JSON with: `merges` (groups of ids "
    "that mean the SAME thing - keep them as small clusters, antonyms are NOT merges), "
    "`polarity_fixes` (id -> corrected polarity pro/con/neutral, judged for the pos form in "
    "isolation relative to the claim), and `nonatomic` (ids whose pos or neg is not a single "
    "simple fact). Also return `tier_fixes` (id -> corrected evidential tier 0/1/2, where 0 is a "
    "self-evident base fact, 1 an intermediate proposition, 2 a near-claim conclusion) for any id "
    "whose tier is clearly wrong. Also return `axiom_fixes` (id -> true/false): set false for any id "
    "wrongly marked axiomatic that is really an empirical or contestable fact, and true only for an "
    "id that is genuinely incontrovertible (true by definition/logic/math). Be conservative: only "
    "merge true synonyms."
)


def apply_critic(claim, atoms, model, host, temp, api_key):
    listing = "\n".join(f'{i}: pos="{a["pos"]}" | neg="{a["neg"]}" | {a["polarity"]} | tier {a["tier"]}'
                        + (" | AXIOMATIC" if a.get("axiomatic") else "")
                        for i, a in atoms.items() if i != "c0")
    user = f'CLAIM: "{claim}"\nInventory:\n{listing}'
    raw = _ollama_chat(model, CRITIC_SYSTEM, user, host, temp, api_key, schema=CRITIC_SCHEMA)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        data = {}
    for i, pol in (data.get("polarity_fixes") or {}).items():
        if i in atoms and pol in ("pro", "con", "neutral"):
            atoms[i]["polarity"] = pol
    for i, t in (data.get("tier_fixes") or {}).items():
        if i in atoms and i != "c0" and isinstance(t, int) and 0 <= t <= atoms["c0"]["tier"] - 1:
            atoms[i]["tier"] = t
    for i, b in (data.get("axiom_fixes") or {}).items():
        if i in atoms and i != "c0":
            atoms[i]["axiomatic"] = bool(b) and atoms[i].get("tier", 0) == 0
    for i in (data.get("nonatomic") or []):
        atoms.pop(i, None)
    remap = {}
    for group in (data.get("merges") or []):
        group = [g for g in group if g in atoms and g != "c0"]
        if len(group) > 1:
            keep = group[0]
            for g in group[1:]:
                remap[g] = keep
                atoms.pop(g, None)
    return atoms, remap

COMPOSE_SYSTEM = (
    "You build ONE argument from a FIXED inventory of atomic propositions. Use ONLY the given "
    "proposition ids, each as a positive literal (e.g. q3) or a negated literal (e.g. -q3, which "
    "means the proposition's negation). `premises` are literals taken as given; each rule chains "
    "antecedent literals to one consequent literal; `conclusion` MUST be the literal you are told "
    "to reach. Build a connected chain from premises to the conclusion. "
    "Respect the evidential tiers: `premises` must be your lowest-tier starting points, and EVERY "
    "rule must CLIMB - the consequent's tier must be strictly higher than the tier of every one of "
    "its antecedents. Never assert a higher-tier proposition as a premise when it could instead be "
    "derived from lower-tier ones; earn it with a rule. Work upward from tier-0 facts toward the "
    "conclusion. Do not invent new propositions or ids. "
    "For EACH rule set `strict` (true/false). Make a rule strict=true ONLY when the antecedents "
    "force the consequent by necessity - given the antecedents the consequent cannot be false, and "
    "no further fact, exception, or counter-argument could ever break that single step (for "
    "example 'X is a square -> X is a rectangle', 'X is a whale -> X is a mammal', 'N is divisible "
    "by four -> N is even'). Set strict=false for a step that merely usually or plausibly holds and "
    "could be overturned by an exception or stronger evidence ('it is a bird -> it flies' fails for "
    "penguins; 'the source says it -> it is true' can be wrong). The FINAL rule - the one whose "
    "consequent is the claim literal you must reach - MUST be defeasible (strict=false): the claim "
    "is debatable, so it can never be reached by an unbreakable step. Use strict only for "
    "intermediate, definitionally-forced links. Return JSON only."
)

STYLES = [
    "a short direct chain of 2-3 steps",
    "two independent premises converging on the conclusion",
    "a longer chain of 4+ steps",
    "two separate lines of reasoning that both reach the conclusion",
    "a chain that uses a negated literal as a step",
    "an argument from empirical evidence",
    "an argument from definition or conceptual analysis",
    "an argument from expert consensus",
    "an argument from analogy",
]


def _sanitize_composed(arg, atoms):
    raw_prem = arg.get("premises", [])
    if isinstance(raw_prem, (str, dict)):
        raw_prem = [raw_prem]
    prem = [s for p in raw_prem if (s := _aslit(p)) and base(s) in atoms]
    rules = []
    for r in (arg.get("rules", []) or []):
        if not isinstance(r, dict):
            continue
        ants_raw = r.get("antecedents", [])
        if isinstance(ants_raw, (str, dict)):
            ants_raw = [ants_raw]
        ants = [s for a in ants_raw if (s := _aslit(a))]
        con = _aslit(r.get("consequent"))
        if ants and con and base(con) in atoms and all(base(x) in atoms for x in ants):
            strict = bool(r.get("strict", False)) and base(con) != "c0" 
            rules.append({"antecedents": ants, "consequent": con, "strict": strict})
    return prem, rules


def compose(claim, atoms, target, stance, style, model, host, temp, api_key):
    listing = "\n".join(f'{i}: {a["pos"]}  (neg: {a["neg"]}; leaning: {a["polarity"]}; tier: {a["tier"]}'
                        + ("; AXIOMATIC - incontrovertible" if a.get("axiomatic") else "") + ")"
                        for i, a in atoms.items() if i != "c0")
    side = "ASSERT the claim" if stance == "support" else "DENY the claim (reach its negation)"
    user = (f'CLAIM (id c0): "{atoms["c0"]["pos"]}"\nInventory:\n{listing}\n\n'
            f'Build an argument that ends at conclusion literal "{target}" to {side}. '
            f'Prefer {style}. Return JSON only.')
    raw = _ollama_chat(model, COMPOSE_SYSTEM, user, host, temp, api_key, schema=ARG_SCHEMA)
    return _parse_json(raw)

PREF_SCHEMA = {
    "type": "object",
    "properties": {
        "preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stronger": {"type": "string"},
                    "weaker": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["stronger", "weaker"],
            },
        },
    },
    "required": ["preferences"],
}

PREF_SYSTEM = (
    "You are given several DEFEASIBLE rules taken from competing arguments about a claim. Each rule "
    "has an id and a plain-English reading. Some support conflicting conclusions. For pairs that "
    "GENUINELY CONFLICT (one supports a statement, the other supports its opposite, or one would "
    "override the other), say which rule should win and why, on ONE of these grounds: specificity "
    "(a more specific rule beats a more general one), reliability (a rule resting on a more "
    "trustworthy or direct basis beats a weaker one), directness (a rule with fewer assumptions "
    "beats a more speculative one), or evidential strength (a rule resting on empirical evidence beats one resting on conjecture or intuition). Return `preferences`: a list of {stronger: id, weaker: id, "
    "reason: short phrase}. Rank ONLY genuinely conflicting pairs; never rank a rule against "
    "itself, never give both directions for a pair, and avoid cycles. Return JSON only."
)


def _filter_prefs(raw_prefs, names):
    seen, pairs, adj = set(), [], {}

    def makes_cycle(s, w):
        stack = [w]
        while stack:
            u = stack.pop()
            if u == s:
                return True
            stack.extend(adj.get(u, []))
        return False

    for p in (raw_prefs or []):
        s, w = _aslit(p.get("stronger")), _aslit(p.get("weaker"))
        if s not in names or w not in names or s == w:
            continue
        if (s, w) in seen or (w, s) in seen:
            continue
        if makes_cycle(s, w):
            continue
        seen.add((s, w))
        adj.setdefault(s, []).append(w)
        pairs.append({"stronger": s, "weaker": w, "reason": (p.get("reason") or "").strip()})
    return pairs


def make_preferences(claim, atoms, drules, model, host, temp, api_key):
    if len(drules) < 2:
        return []
    listing = "\n".join(
        f'{r["name"]}: if ' + " and ".join(gloss(atoms, a) for a in r["antecedents"])
        + f', then {gloss(atoms, r["consequent"])}'
        for r in drules)
    user = f'CLAIM: "{atoms["c0"]["pos"]}"\nDefeasible rules:\n{listing}\n\nReturn JSON only.'
    try:
        raw = _ollama_chat(model, PREF_SYSTEM, user, host, temp, api_key, schema=PREF_SCHEMA)
        data = _parse_json(raw)
    except Exception:
        return []
    prefs = data.get("preferences") if isinstance(data, dict) else (data if isinstance(data, list) else None)
    return _filter_prefs(prefs, {r["name"] for r in drules})


PREM_PREF_SYSTEM = (
    "You are given several ORDINARY PREMISES (plain factual statements) used in competing "
    "arguments about a claim. Rank pairs by CREDIBILITY: which premise is more trustworthy, "
    "better established, or more directly evidenced. Use grounds like: empirical support beats "
    "conjecture, direct observation beats hearsay, well-established fact beats contested "
    "assumption. Return `preferences`: a list of {stronger: premise-text, weaker: premise-text, "
    "reason: short phrase}. Rank ONLY pairs with a genuine credibility difference; never rank a "
    "premise against itself, never give both directions, avoid cycles. Return JSON only."
)


def make_premise_preferences(claim, atoms, premise_lits, model, host, temp, api_key):
    prems = [p for p in premise_lits if p in atoms]
    if len(prems) < 2:
        return []
    listing = "\n".join(f'{p}: {gloss(atoms, p)}' for p in prems)
    user = f'CLAIM: "{atoms["c0"]["pos"]}"\nOrdinary premises:\n{listing}\n\nReturn JSON only.'
    try:
        raw = _ollama_chat(model, PREM_PREF_SYSTEM, user, host, temp, api_key, schema=PREF_SCHEMA)
        data = _parse_json(raw)
    except Exception:
        return []
    text_to_id = {}
    for p in prems:
        text_to_id[gloss(atoms, p).strip().lower()] = p
        text_to_id[p] = p
    remapped = []
    raw_prefs = data.get("preferences") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for pr in (raw_prefs or []):
        if not isinstance(pr, dict):
            continue
        s = text_to_id.get(str(pr.get("stronger", "")).strip().lower())
        w = text_to_id.get(str(pr.get("weaker", "")).strip().lower())
        if s and w:
            remapped.append({"stronger": s, "weaker": w, "reason": (pr.get("reason") or "").strip()})
    return _filter_prefs(remapped, set(prems))

EXCEPTION_SYSTEM = (
    "You supply ONE atomic exception or counter-condition as JSON "
    '{"pos": ..., "neg": ..., "tier": 0}. `pos` is a single present-tense sentence '
    "stating the condition; `neg` states its exact negation (the contradictory, not "
    "a contrary). No conjunctions, no if/then, no hedging. The condition must be a "
    "plausible, on-topic reason in the context you are given."
)
EXC_SCHEMA = {"type": "object",
              "properties": {"pos": {"type": "string"}, "neg": {"type": "string"},
                             "tier": {"type": "integer", "minimum": 0, "maximum": 4}},
              "required": ["pos", "neg"]}


def _attack_dsl(atoms, premises, rules):
    lines = [f"[premise: {p}]" for p in premises]
    for r in rules:
        kw = "strict" if r.get("strict") else "defeasible"
        arrow = "->" if r.get("strict") else "=>"
        label = f' {r["name"]}' if r.get("name") else ""
        lines.append(f"[{kw}{label}: {' AND '.join(r['antecedents'])} {arrow} {r['consequent']}]")
    return "\n".join(lines)


def _negate(lit):
    return lit[1:] if lit.startswith("-") else "-" + lit


def _new_atom_id(atoms):
    i = 1
    while f"e{i}" in atoms:
        i += 1
    return f"e{i}"


def _ask_exception(context, atoms, model, host, temp, api_key, tier=0, tier_max=2):
    try:
        raw = _ollama_chat(model, EXCEPTION_SYSTEM, context, host, temp, api_key,
                           schema=EXC_SCHEMA)
        pdat = _parse_json(raw)
    except Exception:
        return None
    if not pdat or not isinstance(pdat, dict) or not pdat.get("pos") or not pdat.get("neg"):
        return None
    pos, neg = _san(pdat["pos"]), _san(pdat["neg"])
    if _atomic_problems(pos) or _atomic_problems(neg):
        return None
    aid = _new_atom_id(atoms)
    tr = pdat.get("tier", tier)
    atoms[aid] = {"pos": pos, "neg": neg, "polarity": "neutral",
                  "tier": min(tr if isinstance(tr, int) and tr >= 0 else tier, tier_max),
                  "axiomatic": False}
    return aid


def _flip_gate(target_dsl, target_concl, attack_dsl, attack_concl):
    try:
        v = ASPICVerifier.from_dsl(target_dsl + "\n" + attack_dsl,
                                   ordering="last_link_elitist")
    except Exception:
        return False, None
    if not v.is_consistent() or not v.check_invariants().ok:
        return False, None
    after = v.status(target_concl)
    if after == JUSTIFIED:
        return False, None
    if v.status(attack_concl) == "UNSATISFIABLE":
        return False, None
    return True, after


def _next_rule_name(counters, strict):
    key = "s" if strict else "d"
    counters[key] += 1
    return f"{key}{counters[key]}"


def generate_attacks(claim, atoms, kb, counters, want, derived_ok,
                     model, host, temp, api_key, tier_max=2, verbose=True):
    out = []
    targets = [(s, i, a) for s in ("support", "disclaim") for i, a in enumerate(kb[s])]
    if not targets:
        return out
    ti = 0
    for atype in ("rebut", "undermine", "undercut"):
        made, tries = 0, 0
        while made < want.get(atype, 0) and tries < 8 * max(1, want.get(atype, 0)):
            tries += 1
            stance, idx, arg = targets[ti % len(targets)]; ti += 1
            target_dsl, tconcl = arg["dsl"], arg["conclusion"]
            prem, rules, variant, new_atom = [], [], "asserted", None

            if atype == "rebut":
                pts = [r["consequent"] for r in arg["rules"] if not r.get("strict")]
                if not pts:
                    continue
                m = pts[tries % len(pts)]
                point, aconcl = m, _negate(m)
                if derived_ok and tier_of(atoms, m) >= 1:
                    ctx = (f'CLAIM: "{claim}". An argument concludes: "{gloss(atoms, m)}". '
                           f"Give one atomic condition that, if true, directly supports "
                           f"the OPPOSITE.")
                    new_atom = _ask_exception(ctx, atoms, model, host, temp, api_key,
                                              tier=max(0, tier_of(atoms, m) - 1),
                                              tier_max=tier_max)
                    if not new_atom:
                        continue
                    prem = [new_atom]
                    rules = [{"antecedents": [new_atom], "consequent": aconcl,
                              "strict": False, "name": _next_rule_name(counters, False)}]
                    variant = "derived"
                else:
                    if tier_of(atoms, m) > 1:
                        continue
                    prem = [aconcl]

            elif atype == "undermine":        
                pts = list(arg.get("premises") or [])
                if not pts:
                    continue
                pr = pts[tries % len(pts)]
                point, aconcl = pr, _negate(pr)
                prem = [aconcl]

            else: 
                dks = [r["name"] for r in arg["rules"]
                       if not r.get("strict") and r.get("name")]
                if not dks:
                    continue
                dk = dks[tries % len(dks)]
                r = next(r for r in arg["rules"] if r.get("name") == dk)
                rtxt = ("if " + " and ".join(gloss(atoms, a) for a in r["antecedents"])
                        + ", then " + gloss(atoms, r["consequent"]))
                ctx = (f'CLAIM: "{claim}". A default rule says: "{rtxt}". Give one atomic '
                       f"EXCEPTION condition under which this rule should NOT apply "
                       f"(the inference is blocked, not the conclusion denied). "
                       f"State a concrete circumstance, not a restatement of the "
                       f"opposite conclusion.")
                new_atom = _ask_exception(ctx, atoms, model, host, temp, api_key,
                                          tier=0, tier_max=tier_max)
                if not new_atom:
                    continue
                point, aconcl, variant = dk, "-" + dk, "derived"
                prem = [new_atom]
                rules = [{"antecedents": [new_atom], "consequent": aconcl,
                          "strict": False, "name": _next_rule_name(counters, False)}]

            attack_dsl = _attack_dsl(atoms, prem, rules)
            ok, after = _flip_gate(target_dsl, tconcl, attack_dsl, aconcl)
            if not ok:
                if new_atom:                    
                    atoms.pop(new_atom, None)
                continue
            out.append({"type": atype, "variant": variant,
                        "target_stance": stance, "target_index": idx,
                        "target_point": point, "conclusion": aconcl,
                        "premises": prem, "rules": rules, "dsl": attack_dsl,
                        "readable": render_readable(atoms, prem, rules, aconcl),
                        "effect": {"before": JUSTIFIED, "after": after}})
            made += 1
            if verbose:
                print(f"    [attack] {atype}/{variant} on {stance}#{idx} @ {point} -> {after}")
    return out


def compute_metrics(kb):
    depth = _arg_depth
    args = kb["support"] + kb["disclaim"]
    return {
        "max_depth": max((depth(a) for a in args), default=0),
        "n_atoms": len(kb["atoms"]) - 1,
        "n_rules": sum(len(a["rules"]) for a in args),
        "n_attacks": {t: sum(1 for a in kb.get("attacks", []) if a["type"] == t)
                      for t in ("rebut", "undermine", "undercut")},
        "n_derived_attacks": sum(1 for a in kb.get("attacks", [])
                                 if a["variant"] == "derived"),
        "n_prefs": len(kb.get("preferences", [])),
    }


def _arg_depth(arg):
    cons = {r["consequent"]: r for r in arg["rules"]}
    def d(lit, seen=()):
        r = cons.get(lit)
        if not r or lit in seen:
            return 0
        return 1 + max((d(a, seen + (lit,)) for a in r["antecedents"]), default=0)
    return d(arg["conclusion"])


def generate_kb(claim, n_each, model, host, temp, api_key, max_attempts, critic, verbose,
                strict_tiers=False, profile="max"):
    prof = PROFILES[profile]
    n_each = n_each if n_each else prof.get("n_each", 7)
    atoms = build_inventory(claim, model, host, max(0.1, temp - 0.3), api_key,
                            n_props=prof["n_props"], tier_max=prof["tier_max"])
    if not atoms:
        return None
    if critic:
        atoms, _ = apply_critic(claim, atoms, model, host, 0.1, api_key)
    kb = {"claim": claim, "claim_atom": "c0", "atoms": atoms, "support": [], "disclaim": []}
    for stance in ("support", "disclaim"):
        target = "c0" if stance == "support" else "-c0"
        seen, attempts = set(), 0
        while len(kb[stance]) < n_each and attempts < max_attempts:
            style = STYLES[attempts % len(STYLES)]
            attempts += 1
            try:
                arg = compose(claim, atoms, target, stance, style, model, host,
                              temp + 0.1 * (attempts // len(STYLES)), api_key)
            except Exception as ex:
                if verbose:
                    print(f"    [{stance}] attempt {attempts}: rejected (compose error: {ex})")
                continue
            if not arg:
                continue
            prem, rules = _sanitize_composed(arg, atoms)
            if not rules_climb(atoms, rules, strict=strict_tiers):
                if verbose:
                    print(f"    [{stance}] attempt {attempts}: rejected "
                          f"(tier {'not climbing' if strict_tiers else 'descends'})")
                continue
            dsl = build_dsl(atoms, prem, rules, target)
            ok, st = verify(dsl, target)
            if not ok:
                if verbose:
                    print(f"    [{stance}] attempt {attempts}: rejected (conclusion {st})")
                continue
            sig = arg_signature(prem, rules, target)
            if sig in seen:
                continue
            seen.add(sig)
            leaves = _leaves(atoms, prem, rules)
            ax = sorted(l for l in leaves if atoms.get(base(l), {}).get("axiomatic"))
            ord_prem = sorted(l for l in leaves if l not in ax)
            kb[stance].append({
                "stance": stance, "conclusion": target, "status": st,
                "premises": ord_prem, "axioms": ax, "rules": rules, "dsl": dsl,
                "readable": render_readable(atoms, prem, rules, target),
            })
            if verbose:
                print(f"    [{stance}] verified {len(kb[stance])}/{n_each}")

    for stance in ("support", "disclaim"):
        target = "c0" if stance == "support" else "-c0"
        depths = [_arg_depth(a) for a in kb[stance]]
        need = []
        if depths and min(depths) > 2:
            need.append((STYLES[0], lambda dd: dd <= 2))    
        if depths and sum(1 for dd in depths if dd >= 4) < 2:
            need.append((STYLES[2], lambda dd: dd >= 4))  
        for style, want in need:
            extra, seen_sigs = 0, {arg_signature(a["premises"] + a["axioms"],
                                                 a["rules"], a["conclusion"])
                                   for a in kb[stance]}
            while extra < max_attempts // 4:
                extra += 1
                try:
                    arg = compose(claim, atoms, target, stance, style, model, host,
                                  temp, api_key)
                except Exception:
                    continue
                if not arg:
                    continue
                prem, rules = _sanitize_composed(arg, atoms)
                if not rules or not rules_climb(atoms, rules, strict=strict_tiers):
                    continue
                dsl = build_dsl(atoms, prem, rules, target)
                ok, st = verify(dsl, target)
                if not ok:
                    continue
                sig = arg_signature(prem, rules, target)
                if sig in seen_sigs:
                    continue
                cand = {"stance": stance, "conclusion": target, "status": st,
                        "premises": sorted(l for l in _leaves(atoms, prem, rules)
                                           if not atoms.get(base(l), {}).get("axiomatic")),
                        "axioms": sorted(l for l in _leaves(atoms, prem, rules)
                                         if atoms.get(base(l), {}).get("axiomatic")),
                        "rules": rules, "dsl": dsl,
                        "readable": render_readable(atoms, prem, rules, target)}
                if want(_arg_depth(cand)):
                    kb[stance].append(cand)
                    if verbose:
                        print(f"    [{stance}] spread top-up: depth {_arg_depth(cand)}")
                    break

    MIN_ARGS_PER_STANCE = 3
    if len(kb["support"]) < MIN_ARGS_PER_STANCE or len(kb["disclaim"]) < MIN_ARGS_PER_STANCE:
        if verbose:
            print(f"    [reject] record not viable: {len(kb['support'])} support / "
                  f"{len(kb['disclaim'])} disclaim (< {MIN_ARGS_PER_STANCE} per stance)")
        return None

    di = si = 0
    drules = []
    for stance in ("support", "disclaim"):
        for arg in kb[stance]:
            for r in arg["rules"]:
                if r["strict"]:
                    si += 1; r["name"] = f"s{si}"
                else:
                    di += 1; r["name"] = f"d{di}"; drules.append(r)
            arg["dsl"] = build_dsl(atoms, arg["premises"] + arg["axioms"], arg["rules"],
                                   arg["conclusion"])
    counters = {"d": di, "s": si}
    kb["attacks"] = generate_attacks(claim, atoms, kb, counters, prof["attacks"],
                                     prof["derived"], model, host, temp, api_key,
                                     tier_max=prof["tier_max"], verbose=verbose)
    drules = drules + [r for a in kb["attacks"] for r in a["rules"] if not r.get("strict")]
    kb["preferences"] = make_preferences(claim, atoms, drules, model, host,
                                          max(0.1, temp - 0.2), api_key) if drules else []
    _prem_lits = sorted({p for stance in ("support", "disclaim") for a in kb[stance]
                         for p in a.get("premises", [])})
    kb["premise_preferences"] = make_premise_preferences(
        claim, atoms, _prem_lits, model, host, max(0.1, temp - 0.2), api_key) if _prem_lits else []
    kb["profile"] = profile
    kb["generator"] = {"model": model, "temperature": temp, "schema": 2}
    kb["metrics"] = compute_metrics(kb)
    return kb

def selftest():
    atoms = {
        "c0": {"pos": "machines can have consciousness",
               "neg": "machines cannot have consciousness", "polarity": "pro", "tier": 3},
        "q1": {"pos": "consciousness comes from information processing",
               "neg": "consciousness does not come from information processing",
               "polarity": "pro", "tier": 1},
        "q2": {"pos": "machines perform information processing",
               "neg": "machines do not perform information processing",
               "polarity": "neutral", "tier": 0},
        "q3": {"pos": "consciousness requires biological tissue",
               "neg": "consciousness does not require biological tissue",
               "polarity": "con", "tier": 1},
        "q4": {"pos": "machines are made of biological tissue",
               "neg": "machines are not made of biological tissue",
               "polarity": "neutral", "tier": 0},
    }
    support = {"premises": ["q1", "q2"],
               "rules": [{"antecedents": ["q1", "q2"], "consequent": "c0"}],
               "conclusion": "c0"}
    disclaim = {"premises": ["q3", "-q4"],
                "rules": [{"antecedents": ["q3", "-q4"], "consequent": "-c0"}],
                "conclusion": "-c0"}
    for nm, arg, tgt in (("support", support, "c0"), ("disclaim", disclaim, "-c0")):
        dsl = build_dsl(atoms, arg["premises"], arg["rules"], tgt)
        ok, st = verify(dsl, tgt)
        print(f"[{nm}] conclusion={tgt} status={st} verified={ok}")
        print("  symbolic:", dsl.replace("\n", " | "))
        print(render_readable(atoms, arg["premises"], arg["rules"], tgt))
        print()
    dsl = (build_dsl(atoms, support["premises"], support["rules"], "c0") + "\n"
           + build_dsl(atoms, disclaim["premises"], disclaim["rules"], "-c0"))
    v = ASPICVerifier.from_dsl(dsl)
    print("combined: c0 =", v.status("c0"), "| -c0 =", v.status("-c0"))
    print("gloss(q1)  =", gloss(atoms, "q1"))
    print("gloss(-q1) =", gloss(atoms, "-q1"))
    good = [{"antecedents": ["q1", "q2"], "consequent": "c0"}]     
    flat = [{"antecedents": ["q2"], "consequent": "q4"}]          
    inv = [{"antecedents": ["c0"], "consequent": "q2"}]            
    print("default gate (no descent): climbing=", rules_climb(atoms, good),
          "| flat=", rules_climb(atoms, flat), "| descending=", rules_climb(atoms, inv))
    print("strict gate (must climb):  climbing=", rules_climb(atoms, good, strict=True),
          "| flat=", rules_climb(atoms, flat, strict=True),
          "| descending=", rules_climb(atoms, inv, strict=True))

    A = {
        "c0": {"pos": "the figure is a polygon", "neg": "the figure is not a polygon",
               "polarity": "pro", "tier": 3, "axiomatic": False},
        "q1": {"pos": "the figure is a square", "neg": "the figure is not a square",
               "polarity": "pro", "tier": 0, "axiomatic": False},
        "q2": {"pos": "a square is a rectangle", "neg": "a square is not a rectangle",
               "polarity": "neutral", "tier": 0, "axiomatic": True}, 
        "q3": {"pos": "the figure is a rectangle", "neg": "the figure is not a rectangle",
               "polarity": "pro", "tier": 1, "axiomatic": False},
    }
    prem = ["q1"]
    rules = [{"antecedents": ["q1"], "consequent": "q3", "strict": True, "name": "s1"},
             {"antecedents": ["q3"], "consequent": "c0", "strict": False, "name": "d1"}]
    dsl = build_dsl(A, prem, rules, "c0")
    ok, st = verify(dsl, "c0")
    print("\n[axiom+strict] dsl:")
    print("  " + dsl.replace("\n", "\n  "))
    print("  conclusion c0 status =", st, "verified =", ok)
    print(render_readable(A, prem, rules, "c0"))

    names = {"d1", "d2", "d3"}
    raw = [{"stronger": "d1", "weaker": "d2", "reason": "specificity"},
           {"stronger": "d1", "weaker": "d1"},          
           {"stronger": "d2", "weaker": "d1"},           
           {"stronger": "d2", "weaker": "d3", "reason": "reliability"},
           {"stronger": "d3", "weaker": "d1"},              
           {"stronger": "d9", "weaker": "d2"}]          
    kept = _filter_prefs(raw, names)
    print("\n[pref filter] kept:", [(p["stronger"], p["weaker"]) for p in kept],
          "(expected d1>d2, d2>d3)")
    print("selftest done.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims")
    ap.add_argument("--out", default="kb.json")
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--api-key", default=os.environ.get("OLLAMA_API_KEY"))
    ap.add_argument("--n-each", type=int, default=None,
                    help="arguments per stance (default: the max profile's 7)")
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--strict-tiers", action="store_true",
                    help="require every rule to strictly climb in tier (default: forbid descent)")
    ap.add_argument("--extend", metavar="KB_JSON", default=None,
                    help="upgrade an EXISTING kb.json: run only the typed-attack phase "
                         "(+provenance +metrics) per record; write to --out (e.g. kb_v2.json)")
    ap.add_argument("--extend-attacks", default="rebut=1,undermine=1,undercut=1",
                    help="attack counts per record in --extend mode")
    ap.add_argument("--extend-derived", action="store_true",
                    help="allow model-generated derived attacks in --extend mode")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.claims:
        (args.extend or ap.error("--claims is required (or use --selftest / --extend)"))

    if args.extend:
        want = {k: int(v) for k, v in
                (pair.split("=") for pair in args.extend_attacks.split(","))}
        records = json.load(open(args.extend, encoding="utf-8"))
        for i, kb in enumerate(records, 1):
            atoms = kb["atoms"]
            di = sum(1 for s in ("support", "disclaim") for a in kb[s]
                     for r in a["rules"] if not r.get("strict"))
            si = sum(1 for s in ("support", "disclaim") for a in kb[s]
                     for r in a["rules"] if r.get("strict"))
            tmax = max((v.get("tier", 0) for k, v in atoms.items() if k != "c0"), default=2)
            kb["attacks"] = generate_attacks(kb["claim"], atoms, kb, {"d": di, "s": si},
                                             want, args.extend_derived, args.model,
                                             args.host, args.temperature, args.api_key,
                                             tier_max=max(tmax, 2), verbose=not args.quiet)
            kb["profile"] = kb.get("profile", "standard")
            kb.setdefault("generator", {"schema": 2})["extended_by"] = args.model
            kb["metrics"] = compute_metrics(kb)
            print(f"[{i}/{len(records)}] {kb['claim'][:58]} -> {len(kb['attacks'])} attacks "
                  f"({', '.join(a['type'] for a in kb['attacks'])})")
            json.dump(records, open(args.out, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
        print(f"\nDone. {len(records)} records extended -> {args.out}")
        return

    from generate_argumentations import read_claims
    claims = dedupe_claims(read_claims(args.claims))
    if args.limit:
        claims = claims[:args.limit]
    out = []
    for i, claim in enumerate(claims, 1):
        print(f"[{i}/{len(claims)}] {claim[:60]}")
        kb = generate_kb(claim, args.n_each, args.model, args.host, args.temperature,
                         args.api_key, args.max_attempts, not args.no_critic,
                         verbose=not args.quiet, strict_tiers=args.strict_tiers)
        if kb:
            out.append(kb)
            json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            nax = sum(1 for a in kb["atoms"].values() if a.get("axiomatic"))
            nstrict = sum(1 for s in ("support", "disclaim") for arg in kb[s]
                          for r in arg["rules"] if r.get("strict"))
            print(f"    -> {len(kb['atoms'])-1} atoms ({nax} axiomatic), "
                  f"{len(kb['support'])} support, {len(kb['disclaim'])} disclaim, "
                  f"{nstrict} strict rules, {len(kb.get('attacks', []))} attacks, "
                  f"{len(kb.get('preferences', []))} preferences")
    print(f"\nDone. {len(out)} knowledge bases -> {args.out}")


if __name__ == "__main__":
    main()
