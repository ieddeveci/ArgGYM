"""Build the v2 *paired-content* status_query taskset (issue #15).

De-confounds the content-vs-symbolic gap by fixing the logical structure and
varying only the rendering. One harvested KB structure yields:

  * Metric 1 (representation gap): one content rendering + its SYMBOLIC twin over
    the SAME ops (atoms anonymized to a0..aN). Same structure -> same gold -> the
    symbolic-content gap isolates representation, with the logical instance held
    constant (unlike v1, where content and symbolic are different instances).
  * Metric 2 (surface variance): up to K content renderings of one structure drawn
    from DISTINCT KB records (found by Weisfeiler-Leman iso-signature + matched
    central-claim gold). Variance across them = robustness-to-phrasing / world-
    knowledge interference.

Every item is a plain `status_query` item querying the central claim c0 and its
negation -c0 (the KB's canonical anchor), so `score_content`, `evals/runner.py`
and `evals/scoring.py` are all reused unchanged. Structure grouping rides in each
entry's metadata (`structure_id`, `rendering_id`, `record_id`, `wl_sig`).

Output: a standard frozen taskset dir under data/tasksets/ via `write_taskset`.
Every emitted item passes the gold self-check (its own gold scores 1.0) or the
build aborts.
"""
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import sys

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_engine import Operation  # noqa: E402
from aspic_gym import (  # noqa: E402
    GYM_ORDERING, ASPICVerifier, JUSTIFIED, OVERRULED, UNDECIDED,
    _atoms, _gloss, _entry, _render_content_theory, _render_symbolic_theory,
    score_content, load_kb,
)
from prompting import (  # noqa: E402
    _intro, _sym_notation, _CONTENT_NOTATION, _ordering_decl, _format_block,
)
from tasks.status_query import _status_kb_theory, _status_task_block  # noqa: E402
from evals.taskset import write_taskset, sha256_file  # noqa: E402

WL_ROUNDS = 4
_FMT = ("In the answer, write one line for each claim in the form `number: status` "
        "(for example `1: justified`). Use only the words justified, overruled, "
        "undecided, or unsatisfiable.")


# ---------------------------------------------------------------- iso-signature
def _contrary(lit: str) -> str:
    return lit[1:] if lit.startswith("-") else "-" + lit


def wl_signature(ops) -> str:
    """Iso-invariant structural signature over abstract ASPIC ops (atom/rule names
    abstracted away). WL color-refinement over a typed graph; see the v2 probe."""
    prem, rules, prefs_r, prefs_p, lits = {}, {}, [], [], set()
    for o in ops:
        if o.kind in ("premise", "axiom"):
            prem[o.content] = o.kind
            lits.add(o.content)
        elif o.kind in ("defeasible", "strict"):
            rules[o.name] = (tuple(o.antecedents), o.consequent, o.kind == "strict")
            lits.update(o.antecedents)
            lits.add(o.consequent)
        elif o.kind == "prefer_rule":
            prefs_r.append((o.stronger, o.weaker))
        elif o.kind == "prefer_premise":
            prefs_p.append((o.stronger, o.weaker))
            lits.update((o.stronger, o.weaker))

    lit_nodes = {("L", l) for l in lits}
    rule_nodes = {("R", n) for n in rules}
    nodes = lit_nodes | rule_nodes
    color = {("L", l): ("lit", prem.get(l)) for l in lits}
    color.update({("R", n): ("rule", s, len(a)) for n, (a, c, s) in rules.items()})

    edges = []
    for n, (ants, con, _s) in rules.items():
        for a in ants:
            edges.append((("L", a), "ant", ("R", n)))
        edges.append((("R", n), "con", ("L", con)))
    for l in lits:
        c = _contrary(l)
        if ("L", c) in lit_nodes:
            edges.append((("L", l), "neg", ("L", c)))
        if l.startswith("-") and l[1:] in rules:
            edges.append((("L", l), "uc", ("R", l[1:])))
    for s, w in prefs_r:
        if ("R", s) in rule_nodes and ("R", w) in rule_nodes:
            edges.append((("R", s), "pref", ("R", w)))
    for s, w in prefs_p:
        if ("L", s) in lit_nodes and ("L", w) in lit_nodes:
            edges.append((("L", s), "ppref", ("L", w)))

    out_adj, in_adj = defaultdict(list), defaultdict(list)
    for s, et, d in edges:
        out_adj[s].append((et, ">", d))
        in_adj[d].append((et, "<", s))

    def h(x):
        return hashlib.blake2b(repr(x).encode(), digest_size=8).hexdigest()

    col = {n: h(color[n]) for n in nodes}
    for _ in range(WL_ROUNDS):
        col = {n: h((col[n], tuple(sorted(
            [(et, dr, col[m]) for (et, dr, m) in out_adj[n]] +
            [(et, dr, col[m]) for (et, dr, m) in in_adj[n]])))) for n in nodes}
    coarse = (len(lits), len(rules), sum(v == "axiom" for v in prem.values()),
              sum(r[2] for r in rules.values()), len(prefs_r), len(prefs_p), len(edges))
    return h((coarse, tuple(sorted(col.values()))))


# ------------------------------------------------------------------- anonymize
def anonymize(ops):
    """Relabel atoms to a0..aN and rule names to dN/sN (by kind), first-appearance
    order. Makes symbolic mode meaning-free (c0/q/e prefixes and d-named strict
    rules leak role) and is an isomorphism, so gold is unchanged.

    Rule-name references inside literals (undercut consequents like '-d1') and in
    prefer_rule are remapped consistently; genuine atoms and rule names live in
    disjoint namespaces so a literal's base disambiguates which map applies.
    """
    rule_names = {o.name for o in ops if o.kind in ("defeasible", "strict")}

    def base(l):
        return l[1:] if l.startswith("-") else l

    # canonical rule names by kind, in first-appearance order
    rmap, nd, ns = {}, 0, 0
    for o in ops:
        if o.kind in ("defeasible", "strict") and o.name not in rmap:
            if o.kind == "strict":
                ns += 1
                rmap[o.name] = f"s{ns}"
            else:
                nd += 1
                rmap[o.name] = f"d{nd}"

    # canonical atom names, first-appearance order (skip rule-name references)
    order, seen = [], set()

    def note(l):
        b = base(l)
        if b in rule_names or b in seen:
            return
        seen.add(b)
        order.append(b)

    for o in ops:
        if o.kind in ("premise", "axiom"):
            note(o.content)
        elif o.kind in ("defeasible", "strict"):
            for a in o.antecedents:
                note(a)
            note(o.consequent)
        elif o.kind == "prefer_premise":
            note(o.stronger)
            note(o.weaker)
    amap = {b: f"a{i}" for i, b in enumerate(order)}

    def rl(l):
        if l is None:
            return None
        neg, b = l.startswith("-"), base(l)
        nb = rmap.get(b, b) if b in rule_names else amap.get(b, b)
        return ("-" if neg else "") + nb

    out = []
    for o in ops:
        if o.kind in ("premise", "axiom"):
            out.append(replace(o, content=rl(o.content)))
        elif o.kind in ("defeasible", "strict"):
            out.append(replace(o, name=rmap[o.name],
                               antecedents=tuple(rl(a) for a in o.antecedents),
                               consequent=rl(o.consequent)))
        elif o.kind == "prefer_rule":
            out.append(replace(o, stronger=rmap.get(o.stronger, o.stronger),
                               weaker=rmap.get(o.weaker, o.weaker)))
        elif o.kind == "prefer_premise":
            out.append(replace(o, stronger=rl(o.stronger), weaker=rl(o.weaker)))
        else:
            out.append(o)
    return out, amap


# ----------------------------------------------------------------- entry build
def _status_of(sm, lit):
    return sm.get(lit) or "UNSATISFIABLE"


def _mk_entry(level, ordering, theory_text, notation, queries, claim_display, ops, mode, meta):
    v = ASPICVerifier.from_operations(ops, ordering=ordering)
    sm = v.status_map()
    gstat = [_status_of(sm, q) for q in queries]
    claims_block = "\n".join(f"  {i + 1}. {claim_display[i]}" for i in range(len(queries)))
    task_block = _status_task_block(theory_text, notation, claims_block, len(queries))
    prompt = (_intro(level) + "\n\n" + _ordering_decl(ordering) + task_block
              + "\n\n" + _format_block(_FMT))
    reference = "\n".join(f"{i + 1}: {s.lower()}" for i, s in enumerate(gstat))
    entry = _entry("status_query", prompt, reference, ops, ordering,
                   queries=queries, gold_statuses=gstat, n_claims=len(queries),
                   mode=mode, claim_display=list(claim_display), **meta)
    return entry, gstat


def content_entry(level, atoms, ops, ordering, meta):
    q = ["c0", "-c0"]
    disp = [_gloss(atoms, l) for l in q]
    return _mk_entry(level, ordering, _render_content_theory(atoms, ops),
                     _CONTENT_NOTATION, q, disp, ops, "content", meta)


def symbolic_entry(level, ops, ordering, meta):
    anon, amap = anonymize(ops)
    q = [amap["c0"], "-" + amap["c0"]]
    return _mk_entry(level, ordering, _render_symbolic_theory(anon),
                     _sym_notation(level), q, q, anon, "symbolic", meta)


# -------------------------------------------------------------------- harvest
def rec_id(rec):
    return (rec.get("claim") or "")[:60]


def harvest(level, draws, seed, ordering):
    """Return {wl_sig: [ {rec_id, atoms, ops, gstat}, ... first per record ]}."""
    rng = random.Random(seed + level)
    sig_recs = defaultdict(dict)   # sig -> {rec_id: item}
    for _ in range(draws):
        res = _status_kb_theory(rng, level, ordering)
        if not res:
            continue
        rec, atoms, ops = res
        if "c0" not in _atoms(ops):
            continue
        try:
            v = ASPICVerifier.from_operations(ops, ordering=ordering)
            if not v.is_consistent():
                continue
            sm = v.status_map()
        except Exception:
            continue
        gstat = (_status_of(sm, "c0"), _status_of(sm, "-c0"))
        sig = wl_signature(ops)
        rid = rec_id(rec)
        sig_recs[sig].setdefault(rid, {"rec_id": rid, "atoms": atoms, "ops": ops, "gstat": gstat})
    return sig_recs


# --------------------------------------------------------------------- build
def build(levels, draws, n_struct, kmax, seed, ordering):
    all_rows = []
    idx = defaultdict(int)   # (mode, level) -> running idx
    summary = {}
    for level in levels:
        sig_recs = harvest(level, draws, seed, ordering)
        # one structure per signature; renderings = distinct records sharing the
        # majority central-claim gold (matched question across phrasings).
        structs = []
        for sig, by_rec in sig_recs.items():
            items = list(by_rec.values())
            best_gstat, _ = Counter(it["gstat"] for it in items).most_common(1)[0]
            renders = [it for it in items if it["gstat"] == best_gstat]
            structs.append((len(renders), sig, renders))
        # Balance the selection: take up to 2/3 of the budget from high-K
        # structures (Metric 2 needs K>=2 for surface variance), then fill the
        # rest by RANDOM sample of the remainder so the representation-gap sample
        # (Metric 1) is not biased toward only the most-replicated structures.
        structs.sort(key=lambda t: t[0], reverse=True)
        n_hi = min(sum(1 for t in structs if t[0] >= 2), (n_struct * 2) // 3)
        chosen = structs[:n_hi]
        rest = structs[n_hi:]
        random.Random(seed * 7 + level).shuffle(rest)
        chosen += rest[: max(0, n_struct - len(chosen))]

        n_pairs = n_multi = 0
        for si, (_k, sig, renders) in enumerate(chosen):
            sid = f"L{level:02d}-S{si:03d}"
            renders = renders[:kmax]
            # content renderings (Metric 2 uses all; Metric 1 uses rendering 0)
            content_gstats = []
            for ri, it in enumerate(renders):
                meta = {"structure_id": sid, "rendering_id": ri,
                        "record_id": it["rec_id"], "wl_sig": sig[:12], "k_renderings": len(renders)}
                entry, gstat = content_entry(level, it["atoms"], it["ops"], ordering, meta)
                if score_content(entry["answer"], entry) < 1.0:
                    raise RuntimeError(f"content gold self-check failed at {sid} r{ri}")
                content_gstats.append(gstat)
                all_rows.append(_row("status_query", "content", level, idx, entry))
            # symbolic twin of rendering 0 (Metric 1 pair)
            meta = {"structure_id": sid, "rendering_id": 0,
                    "record_id": renders[0]["rec_id"], "wl_sig": sig[:12], "k_renderings": len(renders)}
            sentry, sgstat = symbolic_entry(level, renders[0]["ops"], ordering, meta)
            if score_content(sentry["answer"], sentry) < 1.0:
                raise RuntimeError(f"symbolic gold self-check failed at {sid}")
            if sgstat != content_gstats[0]:
                raise RuntimeError(
                    f"paired gold mismatch at {sid}: symbolic {sgstat} != content {content_gstats[0]}")
            all_rows.append(_row("status_query", "symbolic", level, idx, sentry))
            n_pairs += 1
            if len(renders) >= 2:
                n_multi += 1
        summary[f"L{level:02d}"] = {"signatures": len(sig_recs), "structures": len(chosen),
                                    "paired": n_pairs, "multi_rendering(K>=2)": n_multi}
    return all_rows, summary


def _row(task, mode, level, idx, entry):
    i = idx[(mode, level)]
    idx[(mode, level)] += 1
    return {"sample_id": f"{task}__{mode}__L{level:02d}__{i:03d}",
            "task": task, "mode": mode, "level": level, "idx": i,
            "cell_seed": 0, "prompt": entry["question"], "entry": entry}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,3,5")
    ap.add_argument("--draws", type=int, default=6000)
    ap.add_argument("--n-struct", type=int, default=40, help="structures per level")
    ap.add_argument("--kmax", type=int, default=3, help="max content renderings per structure")
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--name", default="paired-v2")
    a = ap.parse_args()

    levels = [int(x) for x in a.levels.split(",")]
    ordering = GYM_ORDERING
    print(f"harvesting {a.draws} draws/level over levels {levels} ...", flush=True)
    rows, summary = build(levels, a.draws, a.n_struct, a.kmax, a.seed, ordering)

    root = Path(__file__).resolve().parent.parent
    kb_path = root / "kb.json"
    kb_sha = sha256_file(kb_path) if kb_path.exists() else None
    cfg = OmegaConf.create({
        "name": a.name, "task": "status_query", "modes": ["content", "symbolic"],
        "levels": levels, "queries": ["c0", "-c0"], "ordering": ordering,
        "draws_per_level": a.draws, "n_struct": a.n_struct, "kmax": a.kmax,
        "seed": a.seed, "notes": "paired-content v2 (issue #15)",
    })
    out = write_taskset(root / "data" / "tasksets", a.name, rows, cfg, kb_sha)

    n_c = sum(r["mode"] == "content" for r in rows)
    n_s = sum(r["mode"] == "symbolic" for r in rows)
    print(f"\nwrote {len(rows)} rows ({n_c} content, {n_s} symbolic) -> {out}")
    for lv, s in summary.items():
        print(f"  {lv}: {s}")
    print(f"taskset_id: {out.name}")


if __name__ == "__main__":
    main()
