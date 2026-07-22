from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple
from aspic_engine import Operation, contrary, JUSTIFIED, OVERRULED, UNDECIDED, UNSATISFIABLE
from aspic_dsl import parse_dsl, sanitize_statement
from aspic_api import ASPICVerifier
import aspic_content as _content

ALL_STATUSES = [JUSTIFIED, OVERRULED, UNDECIDED, UNSATISFIABLE]
_RULE_NAME = re.compile(r"^-?[ds]\d+$")
GYM_ORDERING = "last_link_elitist"

import levels as difficulty

ATTACK_CHAIN_LEVEL = difficulty.FEATURES["attack_chains"]
MAX_LEVEL = difficulty.MAX_LEVEL
N_CONTENT_BANDS = difficulty.ANCHOR_MAX

def op_to_dict(op: Operation) -> dict:
    return {"kind": op.kind, "content": op.content, "name": op.name,
            "antecedents": list(op.antecedents), "consequent": op.consequent,
            "stronger": op.stronger, "weaker": op.weaker}


def dict_to_op(d: dict) -> Operation:
    return Operation(kind=d["kind"], content=d.get("content"), name=d.get("name"),
                     antecedents=tuple(d.get("antecedents") or ()),
                     consequent=d.get("consequent"),
                     stronger=d.get("stronger"), weaker=d.get("weaker"))


def ops_to_dicts(ops: List[Operation]) -> List[dict]:
    return [op_to_dict(o) for o in ops]


def dicts_to_ops(ds: List[dict]) -> List[Operation]:
    return [dict_to_op(d) for d in ds]


def render_dsl(ops: List[Operation]) -> str:
    out = []
    for o in ops:
        if o.kind in ("premise", "axiom"):
            out.append(f"[{o.kind}: {o.content}]")
        elif o.kind == "defeasible":
            out.append(f"[defeasible: {' AND '.join(o.antecedents)} => {o.consequent}]")
        elif o.kind == "strict":
            out.append(f"[strict: {' AND '.join(o.antecedents)} -> {o.consequent}]")
        elif o.kind == "prefer_rule":
            out.append(f"[prefer_rule: {o.stronger} > {o.weaker}]")
        elif o.kind == "prefer_premise":
            out.append(f"[prefer_premise: {o.stronger} > {o.weaker}]")
    return "\n".join(out)


@dataclass
class TheoryConfig:
    n_atoms: int = 7
    n_premises: int = 3
    p_axiom: float = 0.2
    n_rules: int = 7
    p_strict: float = 0.15
    max_arity: int = 2
    p_conflict: float = 0.4
    p_undercut: float = 0.2
    n_pref: int = 2
    layers: int = 3
    chain_depth: int = 0           
    ordering: str = GYM_ORDERING
    require_conflict: bool = True
    min_arguments: int = 3
    max_tries: int = 80
    n_axioms: Optional[int] = None 
    n_strict: Optional[int] = None  

STATUS_LEVELS: Dict[int, TheoryConfig] = difficulty.make_theory_schedule(TheoryConfig)


def task_elements(kind: str, level: int) -> int:
    return difficulty.task_elements(kind, level)


def _config(level_or_cfg) -> TheoryConfig:
    if isinstance(level_or_cfg, TheoryConfig):
        return level_or_cfg
    return STATUS_LEVELS[level_or_cfg]

@dataclass
class Theory:
    operations: List[Operation]
    ordering: str
    _v: Optional[ASPICVerifier] = field(default=None, repr=False)

    def verifier(self) -> ASPICVerifier:
        if self._v is None:
            self._v = ASPICVerifier.from_operations(self.operations, ordering=self.ordering)
        return self._v


def _atoms(ops: List[Operation]) -> List[str]:
    seen = []
    for o in ops:
        lits = []
        if o.content:
            lits.append(o.content)
        lits += list(o.antecedents)
        if o.consequent:
            lits.append(o.consequent)
        for l in lits:
            base = l[1:] if l.startswith("-") else l
            if not _RULE_NAME.match(l) and base not in seen:
                seen.append(base)
    return seen


def _raw_sample(rng: random.Random, cfg: TheoryConfig) -> List[Operation]:
    atoms = [f"a{i}" for i in range(cfg.n_atoms)]
    layer = {a: rng.randint(0, cfg.layers - 1) for a in atoms}
    prem = sorted(atoms, key=lambda a: layer[a])[:cfg.n_premises]
    ops: List[Operation] = []
    for i, p in enumerate(prem):
        if cfg.n_axioms is not None:
            is_ax = i < cfg.n_axioms
        else:
            is_ax = rng.random() < cfg.p_axiom
        ops.append(Operation(kind="axiom" if is_ax else "premise", content=p))
    drules: List[str] = []
    _conflicts: List[tuple] = []      
    didx = 0
    sidx = 0
    strict_left = cfg.n_strict if cfg.n_strict is not None else None
    derivable = list(prem)              
    depth_of = {p: 0 for p in prem}   
    for ridx in range(cfg.n_rules):
        force_strict = strict_left is not None and ridx < strict_left
        if not force_strict and drules and rng.random() < cfg.p_undercut:
            didx += 1
            tgt = rng.choice(drules)
            src = rng.choice(derivable) if derivable else rng.choice(atoms)
            ops.append(Operation(kind="defeasible", name=f"d{didx}",
                                 antecedents=(src,), consequent="-" + tgt))
            drules.append(f"d{didx}")
            continue
        if not derivable:
            continue
        if force_strict:
            negate = False
            fresh = [a for a in atoms if a not in derivable]
            head = rng.choice(fresh) if fresh else rng.choice(atoms)
        else:
            negate = rng.random() < cfg.p_conflict
            if negate:
                head = rng.choice(derivable)    
            else:
                fresh = [a for a in atoms if a not in derivable]
                if fresh:
                    head = rng.choice(fresh)     
                else:
                    negate, head = True, rng.choice(derivable)
        pool = [a for a in derivable if a != head]  
        if not pool:
            continue
        if cfg.chain_depth and not negate and len(pool) > 1:
            pool = sorted(pool, key=lambda a: depth_of.get(a, 0), reverse=True)
            pool = pool[:max(1, len(pool) // 2)]    
        k = rng.randint(1, min(cfg.max_arity, len(pool)))
        ants = tuple(rng.sample(pool, k))
        strict = force_strict or ((strict_left is None) and (rng.random() < cfg.p_strict) and not negate)
        cons = ("-" + head) if negate else head
        if strict:
            sidx += 1
            ops.append(Operation(kind="strict", name=f"s{sidx}", antecedents=ants, consequent=cons))
        else:
            didx += 1
            ops.append(Operation(kind="defeasible", name=f"d{didx}", antecedents=ants, consequent=cons))
            drules.append(f"d{didx}")
        if not negate:
            derivable.append(head)
            depth_of[head] = 1 + max((depth_of.get(a, 0) for a in ants), default=0)
        else:
            _conflicts.append((f"{'s' if strict else 'd'}{sidx if strict else didx}", head))
    if _conflicts:
        rule_map = {o.name: o for o in ops if o.kind in ("defeasible", "strict") and o.name}
        prem_atoms = {o.content for o in ops if o.kind == "premise"}

        def subtree(head_lit):
            rules, prems, stack, seen = set(), set(), [head_lit], set()
            cons_map = {}
            for o in ops:
                if o.kind in ("defeasible", "strict"):
                    cons_map.setdefault(o.consequent, o)
            while stack:
                lit = stack.pop()
                if lit in seen:
                    continue
                seen.add(lit)
                if lit in prem_atoms:
                    prems.add(lit)
                r = cons_map.get(lit)
                if r is not None:
                    if r.kind == "defeasible" and r.name:
                        rules.add(r.name)
                    stack.extend(r.antecedents or ())
            return rules, prems

        emitted_r, emitted_p = set(), set()
        for atk_rule, head in _conflicts[:max(cfg.n_pref, 1)]:
            defender_rules, defender_prems = subtree(head)
            attacker_rules, attacker_prems = subtree("-" + head if not head.startswith("-") else head[1:])
            attacker_rules.add(atk_rule)
            if not (defender_rules or defender_prems) or not (attacker_rules or attacker_prems):
                continue
            winner_first = rng.random() < 0.5
            wR, wP, lR, lP = (defender_rules, defender_prems, attacker_rules, attacker_prems) \
                if winner_first else (attacker_rules, attacker_prems, defender_rules, defender_prems)
            for a in sorted(wR):
                for b in sorted(lR):
                    if a != b and (a, b) not in emitted_r and (b, a) not in emitted_r:
                        emitted_r.add((a, b))
                        ops.append(Operation(kind="prefer_rule", stronger=a, weaker=b))
            for a in sorted(wP):
                for b in sorted(lP):
                    if a != b and (a, b) not in emitted_p and (b, a) not in emitted_p:
                        emitted_p.add((a, b))
                        ops.append(Operation(kind="prefer_premise", stronger=a, weaker=b))
    elif len(drules) >= 2:
        rank_order = drules[:]
        rng.shuffle(rank_order)
        rank = {r: i for i, r in enumerate(rank_order)}    
        pairs = [(drules[i], drules[j]) for i in range(len(drules))
                 for j in range(i + 1, len(drules))]
        rng.shuffle(pairs)
        for a, b in pairs[:cfg.n_pref]:
            s, w = (a, b) if rank[a] < rank[b] else (b, a)
            ops.append(Operation(kind="prefer_rule", stronger=s, weaker=w))
    return ops


def _redundant_rules(ops: List[Operation]) -> bool:
    axiom_lits = {o.content for o in ops if o.kind in ("axiom", "premise")}
    seen = set()
    for o in ops:
        if o.kind not in ("strict", "defeasible"):
            continue
        key = (frozenset(o.antecedents), o.consequent)
        if key in seen:
            return True
        seen.add(key)
        if o.consequent in axiom_lits:       
            return True
    return False


def sample_theory(rng: random.Random, level_or_cfg=2) -> Theory:
    cfg = _config(level_or_cfg)
    for _ in range(cfg.max_tries):
        ops = _raw_sample(rng, cfg)
        th = Theory(ops, cfg.ordering)
        v = th.verifier()
        if not v.is_consistent() or not v.check_invariants().ok:
            continue
        if _redundant_rules(ops):
            continue
        ag = v.attack_graph()
        if len(ag["arguments"]) < cfg.min_arguments:
            continue
        if cfg.require_conflict and not ag["defeats"]:
            continue
        return th
    # Previously this returned a hard-coded 4-op theory, which silently handed a
    # level-15 cell a level-1 item -- gold stayed correct, only the difficulty
    # label lied. Raising lets ASPICDataset.__getitem__ resample, and a genuine
    # exhaustion surfaces as a countable generator skip instead of bad data.
    raise RuntimeError(
        f"sample_theory exhausted {cfg.max_tries} attempts for {cfg}")


CONTENT_KINDS = {"claim_identification", "formalization", "status_query",
                 "perturbation_prediction", "robustness",
                 "attackers_of", "attack", "preference_construction",
                 "enthymeme", "counter_argumentation", "evidence_construction",
                 "ordering_sensitivity"}

TASK_MIN_LEVEL = {}

from prompting import (_INTRO, _SYMBOLIC_NOTATION, _CONTENT_NOTATION,
                       _intro, _sym_notation, _format_block, _EXEMPLARS, exemplar)


def _entry(kind, question, answer, ops, ordering, **meta) -> dict:
    md = {"kind": kind, "source_dataset": "aspic_" + kind,
          "ops": ops_to_dicts(ops), "ordering": ordering}
    md.update(meta)
    return {"question": question, "answer": str(answer), "metadata": md}


def _gloss(atoms, lit):
    a = lit[1:] if lit.startswith("-") else lit
    rec = atoms.get(a)
    if not rec:
        return lit
    return rec["neg"] if lit.startswith("-") else rec["pos"]


def _render_symbolic_theory(ops):
    out = []
    for o in ops:
        if o.kind == "premise":
            out.append(f"[premise: {o.content}]")
        elif o.kind == "axiom":
            out.append(f"[axiom: {o.content}]")
        elif o.kind == "defeasible":
            out.append(f"[defeasible {o.name}: {' AND '.join(o.antecedents)} => {o.consequent}]")
        elif o.kind == "strict":
            out.append(f"[strict {o.name}: {' AND '.join(o.antecedents)} -> {o.consequent}]")
        elif o.kind == "prefer_rule":
            out.append(f"[prefer_rule: {o.stronger} > {o.weaker}]")
        elif o.kind == "prefer_premise":
            out.append(f"[prefer_premise: {o.stronger} > {o.weaker}]")
    return "\n".join(out)


def _rule_index(ops):
    return {o.name: k for k, o in enumerate(
        [o for o in ops if o.kind in ("defeasible", "strict")], 1)}


def _content_lit(atoms, lit, ridx):
    base = lit[1:] if lit.startswith("-") else lit
    if base in ridx:
        return ("-" if lit.startswith("-") else "") + f"Rule {ridx[base]}"
    return _gloss(atoms, lit)


def _render_content_theory(atoms, ops):
    prem = [o for o in ops if o.kind in ("premise", "axiom")]
    rules = [o for o in ops if o.kind in ("defeasible", "strict")]
    prefs = [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
    ridx = _rule_index(ops)

    def show(lit):
        return _content_lit(atoms, lit, ridx)

    out = []
    if prem:
        out.append("Given facts:")
        for o in prem:
            out.append(f"  - {show(o.content)}" + (" (certain)" if o.kind == "axiom" else ""))
    if rules:
        out.append("Rules:")
        for o in rules:
            ants = " and ".join(show(a) for a in o.antecedents)
            link = "then necessarily" if o.kind == "strict" else "then"
            out.append(f"  Rule {ridx[o.name]}: if {ants}, {link} {show(o.consequent)}.")
    if prefs:
        out.append("Preferences:")
        for o in prefs:
            if o.kind == "prefer_premise":
                out.append(f"  - the statement \"{_gloss(atoms, o.stronger)}\" is stronger than "
                           f"the statement \"{_gloss(atoms, o.weaker)}\".")
            else:
                out.append(f"  - Rule {ridx.get(o.stronger, o.stronger)} is stronger than "
                           f"Rule {ridx.get(o.weaker, o.weaker)}.")
    return "\n".join(out)


def _gloss_lit(gl, lit):
    a = lit[1:] if lit.startswith("-") else lit
    g = gl.get(a, {"pos": a, "neg": "-" + a})
    return g["neg"] if lit.startswith("-") else g["pos"]


def _sentence_dsl(ops, gl, neg_as_minus=False):
    def g(lit):
        if neg_as_minus and lit.startswith("-"):
            a = lit[1:]
            return "-" + gl.get(a, {"pos": a})["pos"]
        return _gloss_lit(gl, lit)
    lines = []
    for o in ops:
        if o.kind in ("premise", "axiom"):
            lines.append(f"[{o.kind}: {g(o.content)}]")
        elif o.kind in ("defeasible", "strict"):
            arrow = "->" if o.kind == "strict" else "=>"
            ants = " AND ".join(g(a) for a in o.antecedents)
            lines.append(f"[{o.kind}: {ants} {arrow} {g(o.consequent)}]")
        elif o.kind == "prefer_rule":
            lines.append(f"[prefer_rule: {o.stronger} > {o.weaker}]")
        elif o.kind == "prefer_premise":
            lines.append(f"[prefer_premise: {g(o.stronger)} > {g(o.weaker)}]")
    return "\n".join(lines)


def _components(ops):
    comps = set()
    sig = {}
    for o in ops:
        if o.kind in ("defeasible", "strict"):
            sig[o.name] = f"{o.kind}|{' & '.join(sorted(o.antecedents))}|{o.consequent}"
    for o in ops:
        if o.kind == "premise":
            comps.add(f"premise|{o.content}")
        elif o.kind == "axiom":
            comps.add(f"axiom|{o.content}")
        elif o.kind in ("defeasible", "strict"):
            comps.add(f"rule|{sig[o.name]}")
        elif o.kind == "prefer_rule":              
            comps.add(f"pref|{sig.get(o.stronger, o.stronger)}|>|{sig.get(o.weaker, o.weaker)}")
        elif o.kind == "prefer_premise":             
            comps.add(f"prefprem|{o.stronger}|>|{o.weaker}")
    return comps


def enth_components(ops):
    comps = set()
    for o in ops:
        if o.kind in ("premise", "axiom"):
            comps.add(f"fact|{o.content}")
        elif o.kind in ("defeasible", "strict"):
            comps.add(f"rule|{' & '.join(sorted(o.antecedents))}|{o.consequent}")
        elif o.kind == "prefer_rule":
            comps.add(f"pref|{o.stronger}|{o.weaker}")
        elif o.kind == "prefer_premise":
            comps.add(f"pref_prem|{o.stronger}|{o.weaker}")
    return comps

EVIDENCE_KB_PATH = os.environ.get("ASPIC_KB", "kb.json")
_KB_CACHE: Dict[str, list] = {}

import zlib as _zlib

KB_SPLIT_BUCKETS = {"train": set(range(8)), "validation": {8}, "eval": {9},
                    "all": set(range(10))}
_KB_SPLIT = "all"


def kb_bucket(rec: dict) -> int:
    return _zlib.crc32((rec.get("claim") or "").encode("utf-8")) % 10

_KB_CUSTOM: Optional[Dict[str, set]] = None


def set_kb_custom_partition(partition: Optional[Dict[str, list]]) -> None:
    global _KB_CUSTOM, _KB_SPLIT
    _KB_CUSTOM = None if partition is None else {k: set(v) for k, v in partition.items()}
    _KB_SPLIT = "all"
    reset_kb_caches()


def set_kb_split(split: str) -> None:
    global _KB_SPLIT
    allowed = set(KB_SPLIT_BUCKETS) | set(_KB_CUSTOM or ())
    if split not in allowed:
        raise ValueError(f"split must be one of {sorted(allowed)}, got {split!r}")
    if split != _KB_SPLIT:
        _KB_SPLIT = split
        reset_kb_caches()


def kb_split() -> str:
    return _KB_SPLIT


class KBError(RuntimeError):
    """Raised when the KB is missing or malformed."""


def _lint_kb(kb: list, path: str) -> list:
    ok, dropped = [], 0
    for rec in kb:
        good = True
        for a in rec.get("atoms", {}).values():
            for k in ("pos", "neg"):
                s = sanitize_statement(a.get(k, ""))
                if not s or s.startswith("-"):
                    good = False
                a[k] = s
        if good and rec.get("atoms"):
            ok.append(rec)
        else:
            dropped += 1
    if dropped:
        print(f"[aspic_gym] WARNING: dropped {dropped} malformed KB record(s) from {path}")
    if not ok:
        raise KBError(f"KB at {path!r} contains no usable records")
    return ok


def load_kb(path=None, required: bool = True):
    path = path or EVIDENCE_KB_PATH
    if path not in _KB_CACHE:
        try:
            with open(path, encoding="utf-8") as f:
                _KB_CACHE[path] = _lint_kb(json.load(f), path)
        except KBError:
            raise
        except Exception as e:
            if required:
                raise KBError(
                    f"cannot load KB from {path!r} ({e}). Content-mode tasks and "
                    f"evidence_construction need it; set ASPIC_KB or pass path=.") from e
            _KB_CACHE[path] = []
    recs = _KB_CACHE[path]
    if _KB_SPLIT == "all" or not recs:
        return recs
    if _KB_CUSTOM and _KB_SPLIT in _KB_CUSTOM:
        claims = _KB_CUSTOM[_KB_SPLIT]
        part = [r for r in recs if r.get("claim") in claims]
    else:
        buckets = KB_SPLIT_BUCKETS[_KB_SPLIT]
        part = [r for r in recs if kb_bucket(r) in buckets]
    if not part and required:
        raise KBError(f"KB split {_KB_SPLIT!r} selects no records from {path!r}")
    return part


_CACHE_RESETTERS: List = []


def register_cache_resetter(fn) -> None:
    _CACHE_RESETTERS.append(fn)


def reset_kb_caches():
    _KB_CACHE.clear()
    _BAND_CACHE.clear()
    _ensure_modular_tasks()
    for fn in _CACHE_RESETTERS:
        fn()


_BAND_CACHE: Dict[str, list] = {}


def _n_rules(ops):
    return sum(1 for o in ops if o.kind in ("defeasible", "strict"))


def _pick_by_level(pool, level, rng, key, tag):
    if not pool:
        return None
    if tag is None:
        s = sorted(pool, key=key)
        n = len(s)
        bands = [s[i * n // N_CONTENT_BANDS:(i + 1) * n // N_CONTENT_BANDS] or s for i in range(N_CONTENT_BANDS)]
    else:
        if tag not in _BAND_CACHE:
            s = sorted(pool, key=key)
            n = len(s)
            _BAND_CACHE[tag] = [s[i * n // N_CONTENT_BANDS:(i + 1) * n // N_CONTENT_BANDS] or s
                                for i in range(N_CONTENT_BANDS)]
        bands = _BAND_CACHE[tag]
    band = bands[min(max(level, 1), N_CONTENT_BANDS) - 1]
    return band[rng.randrange(len(band))]


def _kb_gloss_pool():
    key = "_glosspool"
    if key not in _KB_CACHE:
        pool = []
        for rec in load_kb(required=False):
            for aid, a in rec.get("atoms", {}).items():
                if aid != "c0" and a.get("pos"):
                    pool.append(a["pos"])
        seen = set()
        _KB_CACHE[key] = [p for p in pool if not (p in seen or seen.add(p))]
    return _KB_CACHE[key]


def _ev_base(lit):
    return lit[1:] if lit.startswith("-") else lit


def _ev_leaning(atoms, lit):
    pol = atoms.get(_ev_base(lit), {}).get("polarity", "neutral")
    if lit.startswith("-"):
        pol = {"pro": "con", "con": "pro", "neutral": "neutral"}[pol]
    return pol


def _ops_from_args(args, atoms):
    contra = lambda x: x[1:] if x.startswith("-") else "-" + x
    ops = []; seen_f = set(); seen_r = set()
    for arg in args:
        for lit in arg["premises"]:
            if lit in seen_f or contra(lit) in seen_f:
                continue
            seen_f.add(lit); ops.append(Operation(kind="premise", content=lit))
        for lit in arg.get("axioms", []):
            if lit in seen_f or contra(lit) in seen_f:
                continue
            seen_f.add(lit)
            kk = "axiom" if atoms.get(lit.lstrip("-"), {}).get("axiomatic") else "premise"
            ops.append(Operation(kind=kk, content=lit))
        for r in arg["rules"]:
            key = (tuple(r["antecedents"]), r["consequent"])
            if key in seen_r:
                continue
            seen_r.add(key)
            kk = "strict" if r.get("strict") else "defeasible"
            ops.append(Operation(kind=kk, name=r["name"], antecedents=tuple(r["antecedents"]),
                                 consequent=r["consequent"]))
    return ops


def _kb_status_theory(rng, level, ordering=None):
    ordering = ordering or GYM_ORDERING
    kb = load_kb()
    if not kb:
        return None
    tp = difficulty.knob('kb_theory', 'prem', level)
    tr = difficulty.knob('kb_theory', 'rules', level)
    tpref = difficulty.knob('kb_theory', 'prefs', level)
    rec = kb[rng.randrange(len(kb))]
    atoms = rec["atoms"]
    sup = sorted(rec.get("support", []), key=lambda a: len(a["rules"]))
    dis = sorted(rec.get("disclaim", []), key=lambda a: len(a["rules"]))

    def union_rules(args):
        seen = set()
        for a in args:
            for r in a["rules"]:
                seen.add((tuple(r["antecedents"]), r["consequent"]))
        return len(seen)

    if difficulty.gate('kb_theory', 'easy_single', level):
        cand = [a for a in (sup + dis) if len(a["rules"]) <= tr and len(a["premises"]) <= tp] or \
               sorted(sup + dis, key=lambda a: len(a["premises"]) + len(a["rules"]))[:1]
        rng.shuffle(cand)
        chosen = cand[:1]
    else:
        order, si, di = [], 0, 0           
        while si < len(sup) or di < len(dis):
            if si < len(sup): order.append(sup[si]); si += 1
            if di < len(dis): order.append(dis[di]); di += 1
        chosen, prem = [], set()
        for a in order:
            if chosen and union_rules(chosen) >= tr:
                break
            ap = set(a["premises"])
            contra = lambda p: p[1:] if p.startswith("-") else "-" + p
            if any(contra(p) in prem for p in ap):    
                continue
            chosen.append(a); prem |= ap
    if not chosen:
        return None

    allow_ax = difficulty.gate('kb_theory', 'allow_ax', level)
    allow_strict = difficulty.gate('kb_theory', 'allow_strict', level)
    lines, seen, rule_names = [], set(), set()
    for a in chosen:
        facts = [(p, "premise") for p in a["premises"]]
        facts += [(p, "axiom" if allow_ax else "premise") for p in a.get("axioms", [])]
        for p, kw in facts:
            if ("f", p) not in seen:
                seen.add(("f", p)); lines.append(f"[{kw}: {p}]")
        for r in a["rules"]:
            key = ("r", tuple(r["antecedents"]), r["consequent"])
            if key in seen:
                continue
            seen.add(key)
            strict = bool(r.get("strict")) and allow_strict
            kw = "strict" if strict else "defeasible"
            arrow = "->" if strict else "=>"
            nm = r.get("name")
            label = f" {nm}" if nm else ""
            lines.append(f"[{kw}{label}: {' AND '.join(r['antecedents'])} {arrow} {r['consequent']}]")
            if nm:
                rule_names.add(nm)
    if difficulty.gate('kb_theory', 'undercuts', level):
        ucs = [a for a in rec.get("attacks", [])
               if a.get("type") == "undercut" and a.get("target_point") in rule_names]
        if ucs:
            a = ucs[0]
            for pr in a.get("premises", []):
                if ("f", pr) not in seen:
                    seen.add(("f", pr)); lines.append(f"[premise: {pr}]")
            for r in a.get("rules", []):
                key = ("r", tuple(r["antecedents"]), r["consequent"])
                if key not in seen:
                    seen.add(key)
                    label = f' {r["name"]}' if r.get("name") else ""
                    lines.append(f"[defeasible{label}: "
                                 f"{' AND '.join(r['antecedents'])} => {r['consequent']}]")
    ops = list(parse_dsl("\n".join(lines)).operations)

    if allow_strict and tpref > 0 and rule_names:
        present = {o.name for o in ops if o.kind in ("defeasible", "strict")}
        prefs = [p for p in rec.get("preferences", [])
                 if p.get("stronger") in present and p.get("weaker") in present][:tpref]
        if prefs:
            pl = "\n".join(f"[prefer_rule: {p['stronger']} > {p['weaker']}]" for p in prefs)
            ops += list(parse_dsl(pl).operations)
        if "weakest_link" in ordering:
            ordinary = {o.content for o in ops if o.kind == "premise"}
            ppairs = [p for p in rec.get("premise_preferences", [])
                      if p.get("stronger") in ordinary and p.get("weaker") in ordinary]
            if ppairs:
                pl2 = "\n".join(f"[prefer_premise: {p['stronger']} > {p['weaker']}]" for p in ppairs)
                ops += list(parse_dsl(pl2).operations)
    return rec, atoms, ops

def _rename_in_order(ops):
    di = si = 0; name_map = {}; out = []
    for o in ops:
        if o.kind == "defeasible":
            di += 1; nn = f"d{di}"; name_map[o.name] = nn
            out.append(replace(o, name=nn))
        elif o.kind == "strict":
            si += 1; nn = f"s{si}"; name_map[o.name] = nn
            out.append(replace(o, name=nn))
        else:
            out.append(o)
    return out, name_map


_ENTH_SYMS = [chr(c) for c in range(ord("a"), ord("z") + 1)]


def _fresh_syms(rng, n):
    pool = list(_ENTH_SYMS); rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return pool + [f"p{i}" for i in range(n - len(pool))]

_CI_TOKENS = r"justified|overruled|undecided|unsatisfiable|none|premise|axiom|defeasible|strict"
_CI_TOKEN_RE = re.compile(rf"(?i)(?<![A-Za-z])({_CI_TOKENS})(?![A-Za-z])")
_DIRECTIVE_RE = re.compile(r"\[[^\[\]]*\]")
_DIRECTIVE_SPLIT_RE = re.compile(r"\[([A-Za-z_]+):(.*)\]$")


def _norm_syntax(s):
    t = "".join((s or "").split())
    parts = _DIRECTIVE_RE.findall(t)
    if parts and "".join(parts) == t:            
        out = []
        for p in parts:
            m = _DIRECTIVE_SPLIT_RE.match(p)
            out.append("[" + m.group(1).lower() + ":" + m.group(2) + "]" if m else p)
        return "".join(out)
    return _CI_TOKEN_RE.sub(lambda m: m.group(1).lower(), t)

def _answer_region(text):
    s = text or ""
    m = re.search(r"\[\s*answer\s*\]", s, re.IGNORECASE)
    if not m:
        return None
    rest = s[m.end():]
    c = re.search(r"\[\s*/\s*answer\s*\]", rest, re.IGNORECASE)
    if not c:
        return None
    return rest[:c.start()]


_STATUS_WORDS = r"(justified|overruled|undecided|unsatisfiable)"


def _parse_status_block(region, claims):
    res = {}
    for m in re.finditer(r"(?<![\w-])(\d{1,2})\s*[\.\)\:\-]*\s*" + _STATUS_WORDS, region, re.I):
        n = int(m.group(1))
        if 1 <= n <= len(claims):
            res.setdefault(n - 1, m.group(2).upper())
    for line in region.splitlines():
        nums = re.findall(r"(?<![\w-])(\d{1,2})(?![\w])", line)
        sts = re.findall(_STATUS_WORDS, line, re.I)
        if nums and sts:
            n = int(nums[0])
            if 1 <= n <= len(claims):
                res.setdefault(n - 1, sts[-1].upper())
    for i, c in enumerate(claims):
        if i in res:
            continue
        cb = r"(?<![\w-])" + re.escape(c) + r"(?![\w])"
        m = re.search(cb + r"[^A-Za-z]{0,40}?" + _STATUS_WORDS, region, re.I)
        if m:
            res[i] = m.group(1).upper()
    return res


def _prf1(correct: int, n_gold: int, n_pred: int) -> float:
    if n_gold == 0 and n_pred == 0:
        return 1.0
    if correct == 0 or n_pred == 0:
        return 0.0
    prec = correct / n_pred
    rec = correct / n_gold
    return 2 * prec * rec / (prec + rec)


def _count_spurious_status_lines(region: str, n_claims: int) -> int:
    spurious = 0
    for line in region.splitlines():
        if not re.search(_STATUS_WORDS, line, re.I):
            continue
        nums = [int(x) for x in re.findall(r"(?<![\w-])(\d{1,2})(?![\w])", line)]
        in_range = [n for n in nums if 1 <= n <= n_claims]
        if nums and not in_range:
            spurious += 1  
    return spurious


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _jaccard(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    union = pred | gold
    return len(pred & gold) / len(union) if union else 1.0


def _content_to_ops(region, atoms, base, accept_neg_gloss=True):
    # Drop double quotes before matching: the content prompt *displays* statements
    # in quotes ("Logic is independent of human opinion"), so models echo them
    # back as [premise: "..."]. Gold statements are unquoted, so this only ever
    # helps -- norm(gold) is unchanged.
    norm = lambda s: re.sub(r"\s+", " ",
                            re.sub(r'[“”‘’"]', "", s).strip().lower()).rstrip(".")
    posmap = {norm(info["pos"]): a for a, info in atoms.items()}
    negmap = {norm(info["neg"]): "-" + a for a, info in atoms.items()}
    rules = [o for o in base if o.kind in ("defeasible", "strict")]
    rule_by_idx = {i + 1: o.name for i, o in enumerate(rules)}

    def lit(text):
        t = norm(text)
        if t in posmap:
            return posmap[t]
        if t in negmap:
            return negmap[t] if accept_neg_gloss else None
        if t.startswith("not ") and t[4:] in posmap:
            return contrary(posmap[t[4:]])
        if t.startswith("-"):
            inner = t[1:].strip()
            if inner in posmap:
                return contrary(posmap[inner])
            if inner in negmap:
                return contrary(negmap[inner])
            if inner in {o.name for o in base if o.kind in ("defeasible", "strict")}:
                return contrary(inner)                
        m = re.search(r"rule\s*(\d+)", t)
        if m:
            rn = rule_by_idx.get(int(m.group(1)))
            if rn:
                neg = t.lstrip().startswith("-") or re.search(r"\bnot\b|disabl|switch", t)
                return contrary(rn) if neg else rn
        return None

    hits = re.findall(r"\[(premise|axiom|defeasible|strict)\s*:?\s*([^\[\]]*)\]", region, re.I)
    dn = sum(1 for o in base if o.kind == "defeasible")
    sn = sum(1 for o in base if o.kind == "strict")
    base_rule_count = len(rules)
    added_rule_by_idx = {}
    # Maps a label the model wrote on its own added rule ("rule4", "d5", ...) to
    # the rule's canonical name, so a preference or undercut that names that label
    # resolves. Models echo the prompt's "Rule k" style; the gym directive syntax
    # shows dK/sK. The gold leaves added rules unlabelled and names them by ordinal.
    added_rule_by_label = {}
    ops = []
    for kind, body in hits:
        kind = kind.lower()
        if kind in ("premise", "axiom"):
            L = lit(body)
            if L is None:
                return None
            ops.append(Operation(kind=kind, content=L))
        else:
            # Strip an optional label the model added: "[defeasible Rule 4: A => B]"
            # or "[defeasible d5: A => B]" -- the label sits before the first ':',
            # ahead of the rule arrow.
            label = None
            mlab = re.match(r"\s*(rule\s*\d+|[dsr]\d+)\s*:\s*", body, re.I)
            if mlab and ("=>" in body[mlab.end():] or "->" in body[mlab.end():]):
                label = re.sub(r"\s+", "", mlab.group(1)).lower()
                body = body[mlab.end():]
            sep = "=>" if "=>" in body else ("->" if "->" in body else None)
            if sep is None:
                return None
            lhs, rhs = body.split(sep, 1)
            ants = [lit(x) for x in re.split(r"\bAND\b", lhs, flags=re.I)]
            con = lit(rhs)
            if con is None or any(a is None for a in ants):
                return None
            if kind == "defeasible":
                dn += 1; nm = f"d{dn}"
            else:
                sn += 1; nm = f"s{sn}"
            base_rule_count += 1
            added_rule_by_idx[base_rule_count] = nm
            if label:
                added_rule_by_label[label] = nm
            ops.append(Operation(kind=kind, name=nm, antecedents=tuple(ants), consequent=con))

    def rule_name(txt):
        # A label the model gave its own added rule takes priority over ordinal
        # resolution, so "prefer_rule: Rule 4 > Rule 2" points at the rule the
        # model wrote "[defeasible Rule 4: ...]" for, even if its ordinal differs.
        key = re.sub(r"\s+", "", txt.strip()).lower()
        if key in added_rule_by_label:
            return added_rule_by_label[key]
        m = re.search(r"rule\s*(\d+)", txt.lower())
        if m:
            k = int(m.group(1))
            return rule_by_idx.get(k) or added_rule_by_idx.get(k)
        t = txt.strip()
        return t if t in {o.name for o in base if o.kind in ("defeasible", "strict")} else None
    for typ, body in re.findall(r"\[(prefer_rule|prefer_premise)\s*:?\s*([^\[\]]*)\]", region, re.I):
        if ">" not in body:
            continue
        left, right = [x.strip() for x in body.split(">", 1)]
        if typ.lower() == "prefer_rule":
            s, w = rule_name(left), rule_name(right)
            if s and w:
                ops.append(Operation(kind="prefer_rule", stronger=s, weaker=w))
        else:
            s, w = lit(left), lit(right)
            if s and w:
                ops.append(Operation(kind="prefer_premise", stronger=s, weaker=w))
    return ops or None


def _added_ops(md, region):
    base = dicts_to_ops(md["ops"])
    if md.get("mode") == "content":
        return base, _content_to_ops(region, md["atoms"], base)
    counts = {"d": sum(1 for o in base if o.kind == "defeasible"),
              "s": sum(1 for o in base if o.kind == "strict")}
    try:
        return base, list(parse_dsl(region, counters=counts).operations)
    except Exception:
        return base, None

def _lang_atoms(ops) -> List[str]:
    out = []
    for o in ops:
        lits = ([o.content] if o.content else []) + list(o.antecedents) + \
               ([o.consequent] if o.consequent else [])
        for l in lits:
            base = l[1:] if l.startswith("-") else l
            if base and not _RULE_NAME.match(base) and base not in out:
                out.append(base)
    return out


def _semantic_equiv_score(model_ops, gold_ops, ordering) -> float:
    lang = sorted(set(_lang_atoms(gold_ops)) | set(_lang_atoms(model_ops)))
    if not lang:
        return 0.0
    probes = [None] + [l for x in lang for l in (x, contrary(x))]
    matched = 0
    for probe in probes:
        extra = [] if probe is None else [Operation(kind="premise", content=probe)]
        try:
            vg = ASPICVerifier.from_operations(list(gold_ops) + extra, ordering=ordering)
            vm = ASPICVerifier.from_operations(list(model_ops) + extra, ordering=ordering)
            ok = all(vg.status(l) == vm.status(l)
                     for x in lang for l in (x, contrary(x)))
        except Exception:
            ok = False
        matched += ok
    return matched / len(probes)


def _relabel_rule_refs(text: str) -> str:
    """Rewrite "Rule N" rule references to the DSL name of the N-th rule.

    In symbolic formalization the prompt numbers rules "Rule 1, Rule 2, ..." over
    all rules in order, while the DSL (and the gold) name them d1/s1/d2/... So a
    model that writes "[prefer_rule: Rule 2 > Rule 1]" means the 1st and 2nd
    rules -- it just echoed the prompt's numbering instead of the dK labels. Map
    each "Rule N" to the name the N-th rule directive receives, tracking
    defeasible/strict counters exactly as parse_dsl does so the mapping is
    correct even when strict rules (which shift dK numbering) are present.
    """
    pos_to_name = {}
    d = s = pos = 0
    for m in re.finditer(r"\[\s*(defeasible|strict)\b[^\]]*\]", text, re.I):
        pos += 1
        if m.group(1).lower() == "defeasible":
            d += 1; pos_to_name[pos] = f"d{d}"
        else:
            s += 1; pos_to_name[pos] = f"s{s}"

    def repl(mm):
        return pos_to_name.get(int(mm.group(1)), mm.group(0))

    return re.sub(r"\brule\s*(\d+)", repl, text, flags=re.I)


def score_answer(answer: str, entry: dict) -> float:
    md = entry["metadata"]
    kind = md["kind"]

    region = _answer_region(answer)
    if region is None:
        return 0.0

    if kind == "syntax":
        return 1.0 if _norm_syntax(region) == md["target_norm"] else 0.0

    if kind == "ordering_sensitivity":
        gold = {"last": md["last_status"].lower(), "weak": md["weak_status"].lower()}
        pred = {}
        contradicted = set()
        for ln in region.splitlines():
            m = re.match(r"\s*(last[\s-]*link|weak(?:est)?[\s-]*link)\s*[:.\)]\s*([a-z]+)",
                         ln.strip(), re.I)
            if m:
                key = "last" if m.group(1).lower().startswith("last") else "weak"
                st = m.group(2).lower()
                if key in pred and pred[key] != st:
                    contradicted.add(key)
                pred[key] = st
        if not pred:
            return 0.0
        correct = sum(1 for k in ("last", "weak")
                      if k not in contradicted and pred.get(k) == gold[k])
        return correct / 2.0

    if kind == "status_query":
        claims = md["queries"]; gold = md["gold_statuses"]
        res = _parse_status_block(region, claims)
        seen = {}
        contradicted = set()
        for m in re.finditer(r"(?<![\w-])(\d{1,2})\s*[\.\)\:\-]*\s*" + _STATUS_WORDS,
                             region, re.I):
            i = int(m.group(1))
            if 1 <= i <= len(claims):
                st = m.group(2).upper()
                if i - 1 in seen and seen[i - 1] != st:
                    contradicted.add(i - 1)
                seen[i - 1] = st
        correct = sum(1 for i, g in enumerate(gold)
                      if i not in contradicted and res.get(i) == g)
        spurious = _count_spurious_status_lines(region, len(claims))
        return max(0.0, correct - spurious) / max(1, len(claims))

    if kind == "attackers_of":
        if md["mode"] == "symbolic":
            toks = [t for t in re.split(r"[,\s]+", region.strip()) if t]
            if len(toks) == 1 and toks[0].lower() == "none":
                pred, garbage = set(), 0
            else:
                pred = {t for t in toks if re.fullmatch(r"-?[a-z]\d+", t)}
                garbage = sum(1 for t in toks if not re.fullmatch(r"-?[a-z]\d+", t))
            gold = set(md["gold_lits"])
            union = len(pred | gold) + garbage
            return 1.0 if union == 0 else len(pred & gold) / union
        def clean(s):
            s = re.sub(r"\s+", " ", s.strip().lower())
            s = re.sub(r"^\s*[-*\u2022\d]+[.)\]]?\s*", "", s)  
            return s.strip(" .;,")
        gold = {clean(x) for x in md["gold_sents"] if clean(x)}
        distr = {clean(x) for x in md["distractor_sents"] if clean(x)} - gold
        matched_gold, n_pred = set(), 0
        for raw in region.splitlines():
            ln = clean(raw)
            if not ln or ln == "none":
                continue
            n_pred += 1                   
            if ln in gold:
                matched_gold.add(ln)
        return _prf1(len(matched_gold), len(gold), n_pred)

    if kind == "claim_identification":
        toks = [t for t in re.split(r"[,\s]+", region.strip()) if t]
        if len(toks) == 1 and toks[0].lower() == "none":
            nums, garbage = set(), 0
        else:
            nums = {int(t) for t in toks
                    if t.isdigit() and 1 <= int(t) <= md["n_candidates"]}
            garbage = sum(1 for t in toks
                          if not (t.isdigit() and 1 <= int(t) <= md["n_candidates"]))
        gold = set(md["established"])
        union = len(nums | gold) + garbage
        if union == 0:
            return 1.0
        return len(nums & gold) / union

    if kind == "evidence_construction":
        atoms = md["atoms"]; target = md["target"]; stance = md["stance"]
        base_ops = dicts_to_ops(md.get("base_ops") or [])
        if md.get("mode") == "content":
            added = _content_to_ops(region, atoms, base_ops)
        else:
            counts = {"d": sum(1 for o in base_ops if o.kind == "defeasible"),
                      "s": sum(1 for o in base_ops if o.kind == "strict")}
            try:
                added = list(parse_dsl(region, counters=counts).operations)
            except Exception:
                added = None
        if not added:
            return 0.0
        if any(o.kind == "axiom" for o in added):
            return 0.0
        if any(o.kind == "premise" and o.content == target for o in added):
            return 0.0
        try:
            v = ASPICVerifier.from_operations(base_ops + added, ordering=md["ordering"])
        except Exception:
            return 0.0
        if not (v.status(target) == JUSTIFIED and v.is_consistent()):
            return 0.0
        info = v.explain(target)
        pool = set(md["pool"])
        producers = md.get("producers", {})
        used = set()
        for a in info["arguments"]:
            if a["label"] == "IN":
                for p in a["premises"]:
                    if _ev_base(p) not in pool:
                        return 0.0
                used |= set(a["premises"])
        if not used:
            return 0.0
        if len(used) < md.get("min_premises", 1):
            return 0.0
        bad = "con" if stance == "support" else "pro"
        if any(_ev_leaning(atoms, L) == bad for L in used):
            return 0.0
        memo: Dict[str, int] = {}

        def eff_depth(lit, stack=()):
            if lit in memo:
                return memo[lit]
            if lit in stack:
                return 0
            best = None
            for ants in producers.get(lit, []):
                if all(_ev_base(x) in pool for x in ants):
                    d = 1 + max((eff_depth(x, stack + (lit,)) for x in ants), default=0)
                    best = d if best is None else min(best, d)
            memo[lit] = best or 0
            return memo[lit]

        q = [1.0 / (1.0 + eff_depth(L)) for L in used]
        def _works(sub):
            if not sub:
                return False
            try:
                vv = ASPICVerifier.from_operations(base_ops + sub, ordering=md["ordering"])
            except Exception:
                return False
            return vv.is_consistent() and vv.status(target) == JUSTIFIED
        kept = list(added)
        for op in list(added):
            if len(kept) <= 1:
                break
            trial = [o for o in kept if o is not op]
            if _works(trial):
                kept = trial
        econ = len(kept) / len(added)
        return max(0.0, econ * sum(q) / len(q))

    if kind == "enthymeme":
        base, added = _added_ops(md, region)
        if not added:
            return 0.0
        T = md["target"]
        v = ASPICVerifier.from_operations(base, ordering=md["ordering"])
        try:
            for op in added:
                v.fw.apply(op)
        except Exception:
            return 0.0
        if not (v.status(T) == JUSTIFIED and v.is_consistent()):
            return 0.0
        try:
            if ASPICVerifier.from_operations(added, ordering=md["ordering"]).status(T) == JUSTIFIED:
                return 0.0
        except Exception:
            pass
        if md.get("mode") == "content":
            return _f1(enth_components(added), set(md["gold_components"]))
        if len(added) != md.get("n_missing", len(added)):
            return 0.0
        try:
            info = v.explain(T)
        except Exception:
            return 0.0
        winners = [a for a in info["arguments"]
                   if a.get("label") == "IN" and a.get("conclusion") == T]
        if not winners:
            return 0.0
        sub_comps = {c for c in enth_components(added) if not c.startswith("pref")}
        base_comps = enth_components(base)

        def arg_comps(a):
            c = {f"fact|{p}" for p in a.get("premises", [])}
            for r in a.get("defeasible_rules", []) + a.get("strict_rules", []):
                lhs, _, rhs = r.partition("=>") if "=>" in r else r.partition("->")
                c.add(f"rule|{' & '.join(sorted(x.strip() for x in lhs.split('AND')))}|{rhs.strip()}")
            return c
        for a in winners:
            ac = arg_comps(a)
            if sub_comps <= ac and (ac & base_comps):
                return 1.0
        return 0.0

    if kind == "preference_construction":
        base, allops = _added_ops(md, region)
        added = [o for o in (allops or []) if o.kind in ("prefer_rule", "prefer_premise")]
        if not added:
            return 0.0
        v = ASPICVerifier.from_operations(base, ordering=md["ordering"])
        try:
            for op in added:
                v.fw.apply(op)
        except Exception:
            return 0.0
        if not v.is_consistent():
            return 0.0
        if v.status(md["target"]) != JUSTIFIED:
            return 0.0
        def _works(sub):
            if not sub:
                return False
            vv = ASPICVerifier.from_operations(base, ordering=md["ordering"])
            try:
                for op in sub:
                    vv.fw.apply(op)
            except Exception:
                return False
            return vv.is_consistent() and vv.status(md["target"]) == JUSTIFIED
        kept = list(added)
        for op in list(added):
            if len(kept) <= 1:
                break
            trial = [o for o in kept if o is not op]
            if _works(trial):
                kept = trial
        return len(kept) / max(len(allops), 1)

    if kind == "attack":
        base, added = _added_ops(md, region)
        if not added:
            return 0.0
        def _attack_marks(ops_added):
            try:
                vv = ASPICVerifier.from_operations(base, ordering=md["ordering"])
                for op in ops_added:
                    vv.fw.apply(op)
                if not vv.is_consistent():
                    return None
            except Exception:
                return None
            lits = {o.content for o in ops_added if o.kind in ("premise", "axiom")} \
                | {o.consequent for o in ops_added if o.kind in ("defeasible", "strict")}
            op_ = {o.name for o in base if o.kind == "defeasible"}
            oc_ = {o.consequent for o in base if o.kind == "defeasible"}
            opm = {o.content for o in base if o.kind == "premise"}
            r = md["req"]
            if r == "undermine":
                t = any(contrary(l) in opm for l in lits)
            elif r == "rebut":
                t = any(contrary(l) in oc_ for l in lits)
            elif r == "outprefer":
                t = (any(o.kind in ("prefer_rule", "prefer_premise") for o in ops_added)
                     and any(contrary(l) in opm or contrary(l) in oc_ for l in lits))
            else:
                t = any(contrary(l) in op_ for l in lits)
            goal = (vv.status(md["target"]) == OVERRULED) if r == "outprefer" \
                else (vv.status(md["target"]) != JUSTIFIED)
            return (t, goal)

        full_marks = _attack_marks(added)
        econ = 1.0
        if full_marks is not None:
            kept = list(added)
            for op in list(added):
                if len(kept) <= 1:
                    break
                trial = [o for o in kept if o is not op]
                tm = _attack_marks(trial)
                if tm is not None and all(a >= b for a, b in zip(tm, full_marks)):
                    kept = trial
            econ = len(kept) / len(added)
        v = ASPICVerifier.from_operations(base, ordering=md["ordering"])
        try:
            for op in added:
                v.fw.apply(op)
        except Exception:
            return 0.0
        if not v.is_consistent():
            return 0.0
        succeeded = v.status(md["target"]) != JUSTIFIED
        added_lits = {o.content for o in added if o.kind in ("premise", "axiom")} \
            | {o.consequent for o in added if o.kind in ("defeasible", "strict")}
        ord_prem = {o.content for o in base if o.kind == "premise"}
        def_concl = {o.consequent for o in base if o.kind == "defeasible"}
        def_rule = {o.name for o in base if o.kind == "defeasible"}
        req = md["req"]
        if req == "undermine":
            type_ok = any(contrary(l) in ord_prem for l in added_lits)
            attack_lits = {l for l in added_lits if contrary(l) in ord_prem}
        elif req == "rebut":
            type_ok = any(contrary(l) in def_concl for l in added_lits)
            attack_lits = {l for l in added_lits if contrary(l) in def_concl}
        elif req == "outprefer":
            has_pref = any(o.kind in ("prefer_rule", "prefer_premise") for o in added)
            has_atk = any(contrary(l) in ord_prem or contrary(l) in def_concl
                          for l in added_lits)
            type_ok = has_pref and has_atk
            if not type_ok:
                return 0.0
            return econ * (0.3 + 0.7 * bool(v.status(md["target"]) == OVERRULED))
        else:
            type_ok = any(contrary(l) in def_rule for l in added_lits)
            attack_lits = {l for l in added_lits if contrary(l) in def_rule}
        chain_required = md.get("level", 2) >= ATTACK_CHAIN_LEVEL and req in ("rebut", "undercut")
        if not type_ok:
            return 0.0
        score = 0.3
        if chain_required:
            rule_cons = {o.consequent for o in added if o.kind in ("defeasible", "strict")}
            chain_ok = any(l in rule_cons for l in attack_lits)
        else:
            chain_ok = True
        score += 0.2 * bool(chain_ok)
        score += 0.5 * bool(succeeded and chain_ok)
        return econ * score

    if kind == "counter_argumentation":
        base, added = _added_ops(md, region)
        if not added:
            return 0.0
        T = md["target"]; neg = contrary(T)
        if any(o.kind == "axiom" for o in added):
            return 0.0
        if any(o.kind == "strict" for o in added):
            return 0.0
        if any(o.kind == "premise" and o.content == neg for o in added):
            return 0.0
        v = ASPICVerifier.from_operations(base, ordering=md["ordering"])
        try:
            for op in added:
                v.fw.apply(op)
        except Exception:
            return 0.0
        if not (v.status(neg) == JUSTIFIED and v.is_consistent()):
            return 0.0
        def _works(sub):
            if not sub:
                return False
            vv = ASPICVerifier.from_operations(base, ordering=md["ordering"])
            try:
                for op in sub:
                    vv.fw.apply(op)
            except Exception:
                return False
            return vv.is_consistent() and vv.status(neg) == JUSTIFIED
        kept = list(added)
        for op in list(added):
            if len(kept) <= 1:
                break
            trial = [o for o in kept if o is not op]
            if _works(trial):
                kept = trial
        return len(kept) / len(added)

    if kind == "perturbation_prediction":
        n = md["n_claims"]
        gold = {int(k): str(v).lower() for k, v in md["changed_gold"].items()}
        says_none = re.search(r"\bnone\b", region, re.I) is not None
        pred = {}
        contradicted = set()
        spurious = 0
        for ln in region.splitlines():
            m = re.match(r"\s*(\d+)\s*[:.\)]\s*([a-z]+)", ln.strip(), re.I)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < n:
                    st = m.group(2).lower()
                    if idx in pred and pred[idx] != st:
                        contradicted.add(idx)
                    pred[idx] = st
                else:
                    spurious += 1
        if not gold:
            return 1.0 if (not pred and spurious == 0) else 0.0
        if not pred:
            return 0.0                      
        tp = sum(1 for i, g in gold.items()
                 if i not in contradicted and pred.get(i) == g)
        n_pred = len(pred) + spurious
        prec = tp / n_pred
        rec = tp / len(gold)
        return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)

    if kind == "robustness":
        ops = dicts_to_ops(md["ops"]); target = md["target"]
        variant, direction = md["variant"], md["direction"]
        prem = [o.content for o in ops if o.kind == "premise"]
        if md.get("mode") == "content":
            atoms = md["atoms"]
            r = sanitize_statement(region).lower()
            named = [p for p in prem
                     if sanitize_statement(_gloss(atoms, p)).lower() in r]
        else:
            toks = list(dict.fromkeys(re.findall(r"[a-z]\d+", region)))
            named = [p for p in toks if p in prem]
            alien = [t for t in toks if t not in prem]

        def achieves(rm):
            v = ASPICVerifier.from_operations(
                [o for o in ops if not (o.kind == "premise" and o.content in set(rm))],
                ordering=md["ordering"])
            just = v.status(target) == JUSTIFIED
            return just if direction == "reinstate" else not just

        if variant == "set":
            crit = set(md["critical_set"])
            n_alien = len(alien) if md.get("mode") != "content" else 0
            union = len(set(named) | crit) + n_alien
            return 0.0 if union == 0 else len(set(named) & crit) / union
        want = 2 if variant == "pair" else 1
        if md.get("mode") != "content" and alien:
            return 0.0
        if len(named) != want:
            return 0.0
        if not achieves(named):
            return 0.0
        if variant == "pair":
            minimal = not achieves(named[:1]) and not achieves(named[1:])
            return 1.0 if minimal else 0.3
        return 1.0

    if kind == "formalization":
        if md.get("mode") == "content":
            atoms = md.get("atoms")
            if atoms is None:      
                hits = re.findall(r"\[[^\[\]]*\]", region)      
                pred = _components(parse_dsl(" ".join(hits)).operations)
                return _f1(pred, set(md["gold_components"]))
            pred_ops = _content_to_ops(region, atoms, [], accept_neg_gloss=False)
        else:
            hits = re.findall(r"\[[^\[\]]*\]", region)
            text = _relabel_rule_refs(" ".join(hits))
            try:
                pred_ops = list(parse_dsl(text).operations)
            except Exception:
                pred_ops = None
        if not pred_ops:
            return 0.0
        equiv = _semantic_equiv_score(pred_ops, dicts_to_ops(md["ops"]), md["ordering"])
        if equiv > 0:
            return equiv
        return min(0.25, 0.25 * _f1(_components(pred_ops), set(md["gold_components"])))

    raise ValueError(f"unknown kind {kind}")

def validate_entry(entry: dict) -> Tuple[bool, List[str]]:
    md = entry["metadata"]; kind = md["kind"]; reasons: List[str] = []
    ops = dicts_to_ops(md["ops"]); ordering = md["ordering"]
    v = ASPICVerifier.from_operations(ops, ordering=ordering)

    if not v.is_consistent():
        reasons.append("theory_inconsistent")
    if not v.check_invariants().ok:
        reasons.append("invariants_failed")

    try:
        if score_answer(entry["answer"], entry) < 0.999:
            reasons.append("reference_not_perfect")
    except Exception as e:
        reasons.append(f"reference_raised:{e!r}")

    sm = v.status_map()
    try:
        if kind == "status_query":
            claims = md["queries"]; gold = md["gold_statuses"]
            for c, g in zip(claims, gold):
                if v.status(c) != g:
                    reasons.append(f"status_mismatch({c})")
            if len(claims) >= 2 and len(set(gold)) < 2:
                reasons.append("status_batch_degenerate")
        elif kind == "ordering_sensitivity":
            names = {"JUSTIFIED": "justified", "OVERRULED": "overruled",
                     "UNDECIDED": "undecided", "UNSATISFIABLE": "unsatisfiable"}
            ops2 = dicts_to_ops(md["ops"])
            vL = ASPICVerifier.from_operations(ops2, ordering="last_link_elitist")
            vW = ASPICVerifier.from_operations(ops2, ordering="weakest_link_elitist")
            def _nm(vv, lit):
                s = vv.status_map().get(lit)
                return "unsatisfiable" if s is None else names[str(s)]
            if _nm(vL, md["claim"]) != md["last_status"]:
                reasons.append("last_status_mismatch")
            if _nm(vW, md["claim"]) != md["weak_status"]:
                reasons.append("weak_status_mismatch")
            if md["last_status"] == md["weak_status"]:
                reasons.append("not_divergent")            
        elif kind == "attackers_of":
            ag = v.attack_graph()
            concl = {a["id"]: a["conclusion"] for a in ag["arguments"]}
            tids = {a["id"] for a in ag["arguments"] if a["conclusion"] == md["target"]}
            g2 = sorted({concl[e["from"]] for e in ag["defeats"] if e["to"] in tids})
            if not g2:
                reasons.append("attackers_empty")
            elif g2 != sorted(md["gold_lits"]):
                reasons.append("attackers_mismatch")
        elif kind == "claim_identification":
            est2 = [i + 1 for i, l in enumerate(md["universe"]) if sm.get(l) == JUSTIFIED]
            if est2 != list(md["established"]):
                reasons.append("claim_id_mismatch")
        elif kind == "enthymeme":
            if v.status(md["target"]) == JUSTIFIED:
                reasons.append("enthymeme_target_already_justified")
        elif kind in ("attack", "counter_argumentation"):
            if v.status(md["target"]) != JUSTIFIED:
                reasons.append("target_not_justified_in_base")
        elif kind == "perturbation_prediction":
            v_post = ASPICVerifier.from_operations(dicts_to_ops(md["post_ops"]),
                                                   ordering=ordering)
            names = {"JUSTIFIED": "justified", "OVERRULED": "overruled",
                     "UNDECIDED": "undecided", "UNSATISFIABLE": "unsatisfiable"}
            for q, p in zip(md["queries"], md["pre_statuses"]):
                if names[str(v.status(q))] != p:
                    reasons.append(f"pre_status_mismatch({q})")
            for q, p in zip(md["queries"], md["post_statuses"]):
                if names[str(v_post.status(q))] != p:
                    reasons.append(f"post_status_mismatch({q})")
            true_changed = {i for i, (a, b) in enumerate(zip(md["pre_statuses"], md["post_statuses"]))
                            if a != b}
            if set(md["changed"]) != true_changed:
                reasons.append("changed_set_mismatch")
            for i in true_changed:
                if md["changed_gold"].get(str(i), md["changed_gold"].get(i)) != md["post_statuses"][i]:
                    reasons.append(f"changed_gold_mismatch({i})")
        elif kind == "robustness":
            prem = [o.content for o in ops if o.kind == "premise"]
            def _ach(rm):
                v2 = ASPICVerifier.from_operations(
                    [o for o in ops if not (o.kind == "premise" and o.content in set(rm))],
                    ordering=ordering)
                just = v2.status(md["target"]) == JUSTIFIED
                return just if md["direction"] == "reinstate" else not just
            if md["variant"] == "pair":
                g = md["gold_names"]
                if not (_ach(g) and not _ach(g[:1]) and not _ach(g[1:])):
                    reasons.append("gold_pair_not_minimal_critical")
            else:
                if not all(_ach([p]) for p in md["critical_set"]):
                    reasons.append("critical_set_wrong")
                if any(_ach([p]) for p in prem if p not in md["critical_set"]):
                    reasons.append("critical_set_incomplete")
                if len(md["critical_set"]) >= len(prem):
                    reasons.append("all_premises_critical_degenerate")
        elif kind == "formalization":
            if not md.get("gold_components"):
                reasons.append("formalization_empty_gold")
        elif kind == "evidence_construction":
            if len(md.get("pool", [])) < 2:
                reasons.append("evidence_pool_too_small")
    except Exception as e:
        reasons.append(f"validation_raised:{e!r}")

    return (not reasons), reasons

import inspect as _inspect
_ORDERING_FRAMER_CACHE: Dict = {}


def _framer_accepts_ordering(framer) -> bool:
    key = id(framer)
    if key not in _ORDERING_FRAMER_CACHE:
        try:
            _ORDERING_FRAMER_CACHE[key] = "ordering" in _inspect.signature(framer).parameters
        except (TypeError, ValueError):
            _ORDERING_FRAMER_CACHE[key] = False
    return _ORDERING_FRAMER_CACHE[key]


class ASPICDataset:
    def __init__(self, name: str, seed: int = 0, size: int = 100,
                 level: int = 2, with_content: bool = False, validate: bool = True,
                 ordering: Optional[str] = None):
        _ensure_modular_tasks()
        if name not in FRAMERS:
            raise ValueError(f"unknown task {name!r}; choose from {sorted(FRAMERS)}")
        if not 1 <= level <= MAX_LEVEL:
            raise ValueError(f"level must be 1..{MAX_LEVEL}, got {level!r}")
        self.name = name
        self.seed = seed
        self.size = size
        self.level = level
        self.with_content = with_content and name in CONTENT_KINDS
        self.validate = validate
        self.ordering = ordering

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        framer = FRAMERS[self.name]
        for attempt in range(60):
            rng = random.Random((self.seed * 1_000_003 + idx) * 131 + attempt)
            if self.ordering is not None:
                item_ordering = self.ordering
            else:
                item_ordering = "weakest_link_elitist" if rng.random() < 0.5 else "last_link_elitist"
            try:
                if _framer_accepts_ordering(framer):
                    entry = framer(rng, self.with_content, self.level, ordering=item_ordering)
                else:
                    entry = framer(rng, self.with_content, self.level)
            except KBError:
                raise
            except Exception:
                continue       
            if entry is None:
                continue
            if self.validate:
                ok, _reasons = validate_entry(entry)
                if not ok:
                    continue
            entry["metadata"].update({"idx": idx, "seed": self.seed,
                                      "level": self.level,
                                      "mode": "content" if self.with_content else "symbolic",
                                      "task": self.name})
            return entry
        raise RuntimeError(f"could not produce a valid {self.name} task at idx {idx} "
                           f"(level={self.level}, content={self.with_content})")

    def score_answer(self, answer: str, entry: dict) -> float:
        return score_answer(answer, entry)


def create_dataset(name: str, seed: int = 0, size: int = 100, level: int = 2,
                   with_content: bool = False, validate: bool = True,
                   ordering: Optional[str] = None) -> ASPICDataset:
    return ASPICDataset(name, seed=seed, size=size, level=level,
                        with_content=with_content, validate=validate, ordering=ordering)


@dataclass
class Spec:
    name: str
    weight: float = 1.0
    level: int = 2
    with_content: bool = False


class MixtureDataset:
    def __init__(self, specs: List[Spec], seed: int = 0, size: int = 100):
        self.specs = specs
        self.seed = seed
        self.size = size
        self._sub = [ASPICDataset(s.name, seed=seed * 7919 + i, size=size,
                                  level=s.level, with_content=s.with_content)
                     for i, s in enumerate(specs)]

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed * 6271 + idx)
        weights = [s.weight for s in self.specs]
        k = rng.choices(range(len(self.specs)), weights=weights, k=1)[0]
        return self._sub[k][idx]

    def score_answer(self, answer: str, entry: dict) -> float:
        return score_answer(answer, entry)


def create_mixture(specs: List[Spec], seed: int = 0, size: int = 100) -> MixtureDataset:
    return MixtureDataset(specs, seed=seed, size=size)


FRAMERS: Dict[str, Optional[object]] = {
    "status_query": None,
    "claim_identification": None,
    "attackers_of": None,
    "enthymeme": None,
    "attack": None,
    "counter_argumentation": None,
    "preference_construction": None,
    "formalization": None,
    "syntax": None,
    "evidence_construction": None,
}

from tasks.registry import MODULAR_TASKS as _MODULAR_TASKS
_modular_loaded = False


def _ensure_modular_tasks():
    global _modular_loaded
    if _modular_loaded:
        return
    import importlib
    for _name, (_mod, _fn) in _MODULAR_TASKS.items():
        FRAMERS[_name] = getattr(importlib.import_module(_mod), _fn)
    _modular_loaded = True


ALL_TASKS = sorted(set(FRAMERS) | set(_MODULAR_TASKS))
