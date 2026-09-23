"""Nothing cheaper than walking the attack towers may pick out `claim_chain`'s gold.

Every `claim_chain` theory holds several derivations of the claim: the justifying line and
one to three decoys that the theory defeats. Choosing among them should take working out
which attacks succeed. #185 found it did not: the gold was the only derivation with a
negated root, the largest one, the one whose root preference was listed first, the one
whose every attacker was itself attacked, and the one other rules built on, and each of
those picks scored 1.000.

From level 4 each derivation is attacked by exactly one tower -- a rule rebutting one of
its trunk literals, and rules above it each undercutting the one below -- and that tower
alone decides it: even height and the derivation stands, odd and it falls. So the parity
of the towers scores 1.0 by construction, and it is meant to: counting a tower to its top
is the walk the task asks for. The contract is that nothing cheaper than that walk does
better than a random derivation. Below level 4 the line is unattacked and each decoy
falls to one attack, so one step of attack reasoning is the whole task there, as the
README states.

The check lists every argument for the claim the engine builds, renders each as an
answer, scores it with the task's own scorer, and then scores a set of picks against a
uniform pick over the same derivations. A pick that ties several derivations is scored as
the mean over the tie, which is what a solver breaking ties at random expects.

Two kinds of pick must land within one item of random. Surface picks read signs,
preferences, sizes and what other rules build on, and hold at every level.
Shallow-attack picks read one step of the attack graph -- how many directives attack a
derivation, whether any does, whether an axiom contradicts it, whether every attacker is
itself attacked or out-preferred -- and hold from level 4.

Picks read off quantities the generator draws at random per item -- where a derivation
or its root preference is listed, what it is called, which step its tower hits, and from
level 8 whether its tower is the shortest or the longest -- differ from random by
sampling noise, so they are held to `NOISE_SD` standard deviations instead. That bound
has a blind spot, and it is written down here rather than hidden: at 40 items a level
one standard deviation is about three items, so the default run sees a pick that decides
a whole level (the listed-first root preference sat seven or more away) but not one that
decides a quarter of it. The slow run reads 50 seeds a cell to narrow it. Partial walks
live in that gap on purpose: "every attacker's attacker is undefeated" picks towers of
height exactly 2, and from level 8 that is the line whenever its tower is the shortest,
about ten items in 40. Reading two steps of the tower is part of the walk, not a
shortcut around it.

At levels 4 to 7 the towers are 2 against 3, so the shortest tower is the even one:
length there is parity, and the tower-length picks are not held to random.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.spec import ALL_ORDERINGS, LEVELS
from arggym.tasks import claim_chain as cc

#: The standard grid's ten seeds a cell, 40 items a level. An asserted pick either ties
#: every derivation, and then equals random exactly, or it moves the sum by the share of
#: items it decides -- 20 items a level for the root-sign cue of #185.
SEEDS = tuple(range(10))
#: The slow run's seeds for the picks held to a standard-deviation bound.
MANY_SEEDS = tuple(range(50))

#: The first level at which the line itself is attacked.
ATTACKED_FROM = 4
#: The first level at which tower heights overlap, so that length is not parity.
TOWERS_OVERLAP_FROM = 8

#: A shuffled pick scores random in expectation. Four standard deviations keep some 150
#: checks from failing on luck; see the module docstring for what they cannot see.
NOISE_SD = 4.0


def _neg(x: str) -> str:
    return x[1:] if x.startswith("-") else "-" + x


def derivations(item: cc.CCItem) -> List[Dict]:
    """Every argument for the claim, as an answer and as what a solver can read off the
    printed theory: its shape, its place in the listing, and its direct attackers."""
    v = ASPICVerifier.from_operations(list(item.base_ops), ordering=item.ordering)
    rules = {o.name: o for o in item.base_ops if o.kind in ("defeasible", "strict")}
    kind = {o.content: o.kind for o in item.base_ops if o.kind in ("premise", "axiom")}
    axioms = {c for c, k in kind.items() if k == "axiom"}
    prefs = [o for o in item.base_ops if o.kind == "prefer_premise"]
    pref_rank = {o.stronger: i for i, o in enumerate(prefs)}
    stronger = {o.stronger for o in prefs}
    weaker = {o.weaker for o in prefs}
    beats = defaultdict(set)
    for o in prefs:
        beats[o.weaker].add(o.stronger)
    concluding = defaultdict(list)
    for o in rules.values():
        concluding[o.consequent].append(o.name)
    listed = {cc.render_op(o): i for i, o in enumerate(item.base_ops)}

    def tower(name: str):
        """The rules undercutting one another above an attacking rule, itself first."""
        chain = [name]
        while concluding.get("-" + chain[-1]):
            chain.append(concluding["-" + chain[-1]][0])
        return chain

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
        lit, trunk, steps = item.claim, 0, []
        while lit in by_cons:
            steps.append(lit)
            lit, trunk = by_cons[lit].antecedents[0], trunk + 1
        root = lit
        steps.reverse()
        text = "\n".join([f"[{kind[p]}: {p}]" for p in prem] + [cc.render_op(r) for r in seq])

        # Direct attackers: a fact contradicting a literal, a rule concluding its
        # contrary, a rule undercutting one of its rules. One is "answered" when a
        # stated preference puts the attacked premise above it, or a rule undercuts it.
        lits = set(prem) | {r.consequent for r in rs}
        own = {r.name for r in rs}
        attackers, towers, hit = [], [], []
        for x in lits:
            if _neg(x) in kind:
                attackers.append(_neg(x) not in axioms and x in beats[_neg(x)])
            for rn in concluding.get(_neg(x), []):
                attackers.append(bool(concluding.get("-" + rn)))
                towers.append(tower(rn))
                if x in steps:
                    hit.append(steps.index(x))
        for rn in own:
            for an in concluding.get("-" + rn, []):
                attackers.append(bool(concluding.get("-" + an)))
                towers.append(tower(an))
        out.append({
            "text": text,
            "label": label,
            "key": (frozenset(prem), frozenset(own)),
            "root_neg": root.startswith("-"),
            "branch_neg": any(p.startswith("-") for p in prem if p != root),
            "n_neg": sum(p.startswith("-") for p in prem),
            "root_pref": root in stronger,
            "root_not_weaker": root not in weaker,
            "pref_rank": pref_rank.get(root, len(prefs)),
            "n_prem": len(prem), "n_rules": len(rs),
            "n_junctions": sum(len(r.antecedents) > 1 for r in rs),
            "trunk": trunk,
            "fanout": sum(1 for o in rules.values()
                          if o.name not in own and set(o.antecedents) & lits),
            "n_attacks": len(attackers),
            "unattacked": not attackers,
            "no_axiom_contra": not any(_neg(x) in axioms for x in lits),
            "answered": all(attackers),
            "n_live": sum(not ok for ok in attackers),
            "towers": towers,
            "tower": max((len(t) for t in towers), default=0),
            "hit_step": min(hit, default=trunk),
            "root_pos": listed[f"[{kind[root]}: {root}]"],
            "top_pos": listed[cc.render_op(top)],
            "root_name": root.lstrip("-"),
            "top_name": top.name,
        })
    return out


#: Signs, preferences, sizes and what other rules build on. Asserted at every level.
SURFACE = {
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
    "most_fanout": ("max", "fanout"),
}
#: One step of the attack graph. Asserted from `ATTACKED_FROM`.
SHALLOW_ATTACK = {
    "unattacked": ("flag", "unattacked"),
    "fewest_attacks": ("min", "n_attacks"),
    "no_axiom_contra": ("flag", "no_axiom_contra"),
    "answered": ("flag", "answered"),
    "fewest_live": ("min", "n_live"),
}
STRUCTURAL = {**SURFACE, **SHALLOW_ATTACK}
#: Listing order and names, shuffled per item. Asserted within `NOISE_SD` standard
#: deviations at every level.
SHUFFLED = {
    "pref_first": ("min", "pref_rank"),
    "root_first": ("min", "root_pos"),
    "root_last": ("max", "root_pos"),
    "top_first": ("min", "top_pos"),
    "top_last": ("max", "top_pos"),
    "root_name_first": ("min", "root_name"),
    "top_name_first": ("min", "top_name"),
}
#: The step a tower hits, drawn per derivation from `ATTACKED_FROM`. Asserted within
#: `NOISE_SD` standard deviations from there.
TOWER_TARGET = {
    "hit_earliest": ("min", "hit_step"),
    "hit_latest": ("max", "hit_step"),
}
#: A tower's length is fixed by its role -- even for the line, odd for a decoy -- and
#: only which derivation has the shortest or the longest is balanced, from
#: `TOWERS_OVERLAP_FROM`. Asserted within `NOISE_SD` standard deviations from there.
TOWER_LENGTH = {
    "shortest_tower": ("min", "tower"),
    "longest_tower": ("max", "tower"),
}
ALL_PICKS = {**STRUCTURAL, **SHUFFLED, **TOWER_TARGET, **TOWER_LENGTH}


def exact_picks(level: int) -> Dict:
    return SURFACE if level < ATTACKED_FROM else STRUCTURAL


def noisy_picks(level: int) -> Dict:
    out = dict(SHUFFLED)
    if level >= ATTACKED_FROM:
        out.update(TOWER_TARGET)
    if level >= TOWERS_OVERLAP_FROM:
        out.update(TOWER_LENGTH)
    return out


def picks(item: cc.CCItem) -> Dict[str, float]:
    ds = derivations(item)
    scores = [cc.score(d["text"], item).score for d in ds]

    def mean_over(idx):
        return sum(scores[i] for i in idx) / len(idx)

    row = {"random": mean_over(range(len(ds))), "n_cands": len(ds),
           "n_in": sum(d["label"] == "IN" for d in ds)}
    for name, (how, key) in ALL_PICKS.items():
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


def _noisy_off(tab, level) -> Dict[str, float]:
    base = sum(tab["random"])
    sd = sum(p * (1 - p) for p in tab["random"]) ** 0.5
    return {name: round(sum(tab[name]) - base, 2) for name in noisy_picks(level)
            if abs(sum(tab[name]) - base) > NOISE_SD * sd}


@pytest.mark.parametrize("level", LEVELS)
def test_no_shortcut_beats_a_random_derivation(level):
    tab = surface_table(levels=(level,))[level]
    base = sum(tab["random"])
    off = {name: round(sum(tab[name]) - base, 2) for name in exact_picks(level)
           if abs(sum(tab[name]) - base) > 1.0}
    assert not off, (
        f"L{level}: over {len(tab['random'])} items these picks score more than one item "
        f"away from a random derivation (random sums to {base:.2f}): {off}")
    off = _noisy_off(tab, level)
    assert not off, f"L{level}: more than {NOISE_SD} SD from random: {off}"


@pytest.mark.slow
@pytest.mark.parametrize("level", LEVELS)
def test_no_shuffled_pick_beats_a_random_derivation_on_many_seeds(level):
    tab = surface_table(levels=(level,), seeds=MANY_SEEDS)[level]
    off = _noisy_off(tab, level)
    assert not off, (f"L{level}: over {len(tab['random'])} items, more than {NOISE_SD} SD "
                     f"from random: {off}")


@pytest.mark.parametrize("level", LEVELS)
def test_the_gold_is_the_only_justified_derivation(level):
    for ordering in ALL_ORDERINGS:
        for seed in SEEDS:
            it = cc.make_item(level, seed, ordering)
            ds = derivations(it)
            ok = [d for d in ds if d["label"] == "IN"]
            assert len(ds) >= 2 and len(ok) == 1, (level, ordering, seed, len(ds), len(ok))
            assert cc.score(ok[0]["text"], it).score == 1.0
            assert cc.score(it.reference, it).score == 1.0


@pytest.mark.parametrize("level", [lv for lv in LEVELS if lv >= ATTACKED_FROM])
def test_each_derivation_is_decided_by_its_one_tower(level):
    """Delete the top of a derivation's tower and its standing flips: nothing else
    attacking it decides."""
    for ordering in ALL_ORDERINGS:
        for seed in SEEDS:
            it = cc.make_item(level, seed, ordering)
            for d in derivations(it):
                assert len(d["towers"]) == 1, (level, ordering, seed, d["towers"])
                top = d["towers"][0][-1]
                sub = dataclasses.replace(it, base_ops=[
                    o for o in it.base_ops if not (o.kind == "defeasible" and o.name == top)])
                after = {e["key"]: e["label"] for e in derivations(sub)}
                assert (after[d["key"]] == "IN") != (d["label"] == "IN"), (
                    level, ordering, seed, d["label"], after[d["key"]])
