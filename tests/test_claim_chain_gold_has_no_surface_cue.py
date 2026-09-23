"""No feature of a `claim_chain` derivation that ignores attacks may pick out the gold.

Every `claim_chain` theory holds several derivations of the claim: the justifying line and
one to three decoys that the theory defeats. Choosing among them should take reading the
attacks. It did not (#185): from level 4 the gold was the only derivation whose root
premise was negated, and from level 6 it was also the largest, so "take the one with a
`-` premise" or "take the one with the most rules" scored 1.000 on 480 and 400 of the 600
rows of the standard grid.

The check lists every argument for the claim the engine builds, renders each as an
answer, scores it with the task's own scorer, and then scores a set of surface picks
against a uniform pick over the same derivations. A pick that ties several derivations
is scored as the mean over the tie, which is what a solver breaking ties at random
expects. Each structural pick has to land within one item of random at every level.

Listing position and literal names are reported by `surface_table` too, but not asserted
here: they are shuffled per seed, so on a few dozen items they differ from random by
sampling noise and a one-item bound would fail on luck rather than on a cue.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.spec import ALL_ORDERINGS, LEVELS
from arggym.tasks import claim_chain as cc

#: The standard grid's ten seeds a cell, 40 items a level. A structural pick either
#: ties every derivation, and then equals random exactly, or it moves the sum by the
#: share of items it decides -- 20 items a level for the root-sign cue this replaced.
SEEDS = tuple(range(10))


def derivations(item: cc.CCItem) -> List[Dict]:
    """Every argument for the claim, as an answer and as the features a solver can read
    off the printed theory without evaluating a single attack."""
    v = ASPICVerifier.from_operations(list(item.base_ops), ordering=item.ordering)
    rules = {o.name: o for o in item.base_ops if o.kind in ("defeasible", "strict")}
    kind = {o.content: o.kind for o in item.base_ops if o.kind in ("premise", "axiom")}
    stronger = {o.stronger for o in item.base_ops if o.kind == "prefer_premise"}
    weaker = {o.weaker for o in item.base_ops if o.kind == "prefer_premise"}
    listed = {cc.render_op(o): i for i, o in enumerate(item.base_ops)}
    out = []
    for a, label in v.fw.argument_labels().items():
        if str(a.conclusion) != item.claim:
            continue
        prem = sorted(str(p) for p in a.premises)
        rs = [rules[str(r.id)] for r in list(a.defeasible_rules) + list(a.strict_rules)]
        have, seq, todo = set(prem), [], list(rs)
        while todo:
            r = next((r for r in todo if all(x in have for x in r.antecedents)), None)
            if r is None:
                seq += todo
                break
            seq.append(r)
            have.add(r.consequent)
            todo.remove(r)
        by_cons = {r.consequent: r for r in rs}
        top = by_cons[item.claim]
        lit, trunk = item.claim, 0
        while lit in by_cons:
            lit, trunk = by_cons[lit].antecedents[0], trunk + 1
        root = lit
        text = "\n".join([f"[{kind[p]}: {p}]" for p in prem] + [cc.render_op(r) for r in seq])
        out.append({
            "text": text,
            "label": label,
            "root_neg": root.startswith("-"),
            "branch_neg": any(p.startswith("-") for p in prem if p != root),
            "n_neg": sum(p.startswith("-") for p in prem),
            "root_pref": root in stronger,
            "root_not_weaker": root not in weaker,
            "n_prem": len(prem), "n_rules": len(rs),
            "n_junctions": sum(len(r.antecedents) > 1 for r in rs),
            "trunk": trunk,
            "root_pos": listed[f"[{kind[root]}: {root}]"],
            "top_pos": listed[cc.render_op(top)],
            "root_name": root.lstrip("-"),
            "top_name": top.name,
        })
    return out


#: Picks read off the theory's structure: signs, preferences and sizes. Asserted.
STRUCTURAL = {
    "neg_root": ("flag", "root_neg"),
    "neg_branch": ("flag", "branch_neg"),
    "pref_root": ("flag", "root_pref"),
    "root_not_weaker": ("flag", "root_not_weaker"),
    "most_neg": ("max", "n_neg"),
    "most_prem": ("max", "n_prem"),
    "fewest_prem": ("min", "n_prem"),
    "most_rules": ("max", "n_rules"),
    "fewest_rules": ("min", "n_rules"),
    "most_junc": ("max", "n_junctions"),
    "fewest_junc": ("min", "n_junctions"),
    "longest_trunk": ("max", "trunk"),
}
#: Picks read off where a derivation is listed and what it is called. Reported only.
SHUFFLED = {
    "root_first": ("min", "root_pos"),
    "root_last": ("max", "root_pos"),
    "top_first": ("min", "top_pos"),
    "top_last": ("max", "top_pos"),
    "root_name_first": ("min", "root_name"),
    "top_name_first": ("min", "top_name"),
}


def picks(item: cc.CCItem) -> Dict[str, float]:
    ds = derivations(item)
    scores = [cc.score(d["text"], item).score for d in ds]

    def mean_over(idx):
        return sum(scores[i] for i in idx) / len(idx)

    row = {"random": mean_over(range(len(ds))), "n_cands": len(ds)}
    for name, (how, key) in {**STRUCTURAL, **SHUFFLED}.items():
        vals = [d[key] for d in ds]
        if how == "flag":
            idx = [i for i, x in enumerate(vals) if x] or range(len(ds))
        else:
            best = max(vals) if how == "max" else min(vals)
            idx = [i for i, x in enumerate(vals) if x == best]
        row[name] = mean_over(idx)
    return row


def surface_table(levels=LEVELS, seeds=SEEDS) -> Dict[int, Dict[str, List[float]]]:
    tab: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for level in levels:
        for ordering in ALL_ORDERINGS:
            for seed in seeds:
                it = cc.make_item(level, seed, ordering)
                assert it is not None, f"L{level} {ordering} seed {seed} built nothing"
                for k, x in picks(it).items():
                    tab[level][k].append(x)
    return tab


@pytest.mark.parametrize("level", LEVELS)
def test_no_structural_pick_beats_a_random_derivation(level):
    tab = surface_table(levels=(level,))[level]
    base = sum(tab["random"])
    off = {name: round(sum(tab[name]) - base, 2) for name in STRUCTURAL
           if abs(sum(tab[name]) - base) > 1.0}
    assert not off, (
        f"L{level}: over {len(tab['random'])} items these picks score more than one item "
        f"away from a random derivation (random sums to {base:.2f}): {off}")


@pytest.mark.parametrize("level", LEVELS)
def test_the_gold_is_the_only_justified_derivation(level):
    for ordering in ALL_ORDERINGS:
        it = cc.make_item(level, 0, ordering)
        ds = derivations(it)
        ok = [d for d in ds if d["label"] == "IN"]
        assert len(ds) >= 2 and len(ok) == 1, (level, ordering, len(ds), len(ok))
        assert cc.score(ok[0]["text"], it).score == 1.0
        assert cc.score(it.reference, it).score == 1.0
