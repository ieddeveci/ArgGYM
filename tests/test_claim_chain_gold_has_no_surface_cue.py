"""No shortcut short of evaluating the attacks may pick out `claim_chain`'s gold.

Every `claim_chain` theory holds several derivations of the claim: the justifying line and
one to three decoys that the theory defeats. Choosing among them should take working out
which attacks succeed. #185 found it did not: the gold was the only derivation with a
negated root, the largest one, the one whose root preference was listed first, and the
one whose every attacker was itself attacked, and each of those picks scored 1.000.

The check lists every argument for the claim the engine builds, renders each as an
answer, scores it with the task's own scorer, and then scores a set of picks against a
uniform pick over the same derivations. A pick that ties several derivations is scored as
the mean over the tie, which is what a solver breaking ties at random expects.

Two kinds of pick must land within one item of random. Surface picks read signs,
preferences and sizes, and hold at every level. Shallow-attack picks read one step
of the attack graph -- how many directives attack a derivation, whether any does, whether
an axiom contradicts it, whether every attacker is itself attacked or out-preferred --
and hold from level 4. Below that the line is unattacked and each decoy falls to one
unanswered attack, so one step of attack reasoning is the whole task there, as the README
states.

Picks read off quantities the generator draws at random per item -- where a derivation
or its root preference is listed, what it is called, how long its attack tower is and
which step it hits -- differ from random by sampling noise on a few dozen items, so a
one-item bound would fail on luck. They are held to a few standard deviations instead,
which still catches a pick that decides a level: the gold's root preference was listed
first on every row from level 4 until the preferences were shuffled too.
"""
from __future__ import annotations

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

#: The first level at which the line itself is attacked.
ATTACKED_FROM = 4

#: A shuffled pick scores random in expectation, and over 40 items its sum has a
#: standard deviation of about three items. Four of them keep 150 checks from failing
#: on luck, and a pick that decides every item of a level lands six or more away.
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

    def tower(name: str) -> int:
        """How many rules undercut one another above an attacking rule, itself included."""
        n = 1
        while concluding.get("-" + name):
            name, n = concluding["-" + name][0], n + 1
        return n

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
        attackers = []
        for x in lits:
            if _neg(x) in kind:
                answered = _neg(x) not in axioms and x in beats[_neg(x)]
                attackers.append((answered, None, None))
            for rn in concluding.get(_neg(x), []):
                at = steps.index(x) if x in steps else None
                attackers.append((bool(concluding.get("-" + rn)), tower(rn), at))
        for r in rs:
            for rn in concluding.get("-" + r.name, []):
                attackers.append((bool(concluding.get("-" + rn)), tower(rn), None))
        towers = [t for _, t, _ in attackers if t is not None]
        hit = [at for _, _, at in attackers if at is not None]
        out.append({
            "text": text,
            "label": label,
            "root_neg": root.startswith("-"),
            "branch_neg": any(p.startswith("-") for p in prem if p != root),
            "n_neg": sum(p.startswith("-") for p in prem),
            "root_pref": root in stronger,
            "root_not_weaker": root not in weaker,
            "pref_rank": pref_rank.get(root, len(prefs)),
            "n_prem": len(prem), "n_rules": len(rs),
            "n_junctions": sum(len(r.antecedents) > 1 for r in rs),
            "trunk": trunk,
            "n_attacks": len(attackers),
            "unattacked": not attackers,
            "no_axiom_contra": not any(_neg(x) in axioms for x in lits),
            "answered": all(ok for ok, _, _ in attackers),
            "n_live": sum(not ok for ok, _, _ in attackers),
            "tower": max(towers, default=0),
            "hit_step": min(hit, default=trunk),
            "root_pos": listed[f"[{kind[root]}: {root}]"],
            "top_pos": listed[cc.render_op(top)],
            "root_name": root.lstrip("-"),
            "top_name": top.name,
        })
    return out


#: Signs, preferences and sizes. Asserted at every level.
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
#: Drawn at random per item. Asserted within `NOISE_SD` standard deviations.
SHUFFLED = {
    "pref_first": ("min", "pref_rank"),
    "root_first": ("min", "root_pos"),
    "root_last": ("max", "root_pos"),
    "top_first": ("min", "top_pos"),
    "top_last": ("max", "top_pos"),
    "root_name_first": ("min", "root_name"),
    "top_name_first": ("min", "top_name"),
}
#: The attack tower's length and target, drawn at random per derivation from level 4.
#: Below that only the decoys are attacked, so these read the same one step as
#: `SHALLOW_ATTACK`. Asserted within `NOISE_SD` standard deviations from `ATTACKED_FROM`.
SHUFFLED_ATTACK = {
    "shortest_tower": ("min", "tower"),
    "longest_tower": ("max", "tower"),
    "hit_earliest": ("min", "hit_step"),
    "hit_latest": ("max", "hit_step"),
}


def picks(item: cc.CCItem) -> Dict[str, float]:
    ds = derivations(item)
    scores = [cc.score(d["text"], item).score for d in ds]

    def mean_over(idx):
        return sum(scores[i] for i in idx) / len(idx)

    row = {"random": mean_over(range(len(ds))), "n_cands": len(ds),
           "n_in": sum(d["label"] == "IN" for d in ds)}
    for name, (how, key) in {**STRUCTURAL, **SHUFFLED, **SHUFFLED_ATTACK}.items():
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
def test_no_shortcut_beats_a_random_derivation(level):
    tab = surface_table(levels=(level,))[level]
    base = sum(tab["random"])
    exact = SURFACE if level < ATTACKED_FROM else STRUCTURAL
    off = {name: round(sum(tab[name]) - base, 2) for name in exact
           if abs(sum(tab[name]) - base) > 1.0}
    assert not off, (
        f"L{level}: over {len(tab['random'])} items these picks score more than one item "
        f"away from a random derivation (random sums to {base:.2f}): {off}")

    sd = sum(p * (1 - p) for p in tab["random"]) ** 0.5
    noisy = SHUFFLED if level < ATTACKED_FROM else {**SHUFFLED, **SHUFFLED_ATTACK}
    off = {name: round(sum(tab[name]) - base, 2) for name in noisy
           if abs(sum(tab[name]) - base) > NOISE_SD * sd}
    assert not off, (
        f"L{level}: these picks sit more than {NOISE_SD} standard deviations "
        f"({NOISE_SD * sd:.1f} items) from a random derivation: {off}")


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
