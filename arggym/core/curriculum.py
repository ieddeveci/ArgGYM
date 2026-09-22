"""The level-indexed knobs the task builders share.

Junction density, ternary junctions, the per-task junction cap and the four
language profiles: each is a number a level implies and several builders read.
`TASK_PROFILES` is the exception and stays on purpose. It has no reader, but
`docs/dataset-contract.md` records why it is kept rather than dropped: the
profile axis is real and nine tasks branch on it, while two of them take no
profile at all, so the axis is under-built rather than dead (#51).

This module used to also hold `ItemSpec`, an eighteen-field description of what
an item at a level should look like, and `satisfied_by`, a floor on how many
moves a solution must take. Together they read as one difficulty definition the
three `attack_defense` modes answered to. They were not one. `build_attack_item`
was the only caller of `spec_for` and read two of the eighteen fields, both caps
that never bind at or below level 15; `defence` and `mixed` never called it; and
`satisfied_by` had no caller anywhere.

Turning it on instead of removing it was the alternative, and the measurement
rules it out. It was taken against 32 items per mode at levels 3, 6, 9, 12 and
15, which was the whole grid at the time. The required-move floor keeps 0 of 32
`defence` items at all five of them and 0 of 32 `attack_defense` items at levels
9, 12 and 15; adding the dead-end floor drops `attack` at levels 3 and 6 as
well, and `attack` is the mode whose minimum the required floor reproduces digit
for digit at all five. Enforcing the floor empties twelve of those fifteen
(mode, level) cells rather than levelling them.
A floor no builder can meet is one mode's numbers kept somewhere central, so it
is gone and each builder states its own shape where a reader of that builder can
see it (#31).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

ATTACK, DEFENCE, MIXED = "attack", "defence", "attack_defense"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"


JUNCTION_START = 5
JUNCTION_SHARE = 0.20


def junctions_for(level: int, n_rules: int, start: int = JUNCTION_START,
                  share: float = JUNCTION_SHARE, solve: bool = True) -> int:
    if level < start:
        return 0
    if not solve:
        return max(1, round(share * max(1, n_rules)))
    extra = 1.0
    denom = max(0.05, 1.0 - share * extra)
    return max(1, round(share * max(1, n_rules) / denom))


def junction_budget(level: int, cap: int, start: int = JUNCTION_START) -> int:
    if level < start:
        return 0
    span = max(1, 15 - start)
    return max(1, min(cap, round(1 + (level - start) * (cap - 1) / span)))


JUNCTION_CAPS = {
    "status_query": 20,
    "preference_construction": 14,
    "perturbation": 12,
    "formalization": 5,
    "claim_chain": 10,
    "defeat_diagnosis": 9,
    "counter_argument": 8,
    "attack_defense": 10,
    "semantics_query": 2,
}

NEGATED_BRANCH_SHARE = 0.5


def negated_branch(index: int, share: float = NEGATED_BRANCH_SHARE) -> bool:
    step = max(1, round(1 / max(share, 0.01)))
    return (index % step) == 0


TERNARY_FROM_LEVEL = 9
TERNARY_SHARE = 0.4


def wants_ternary(level: int, index: int) -> bool:
    if level < TERNARY_FROM_LEVEL:
        return False
    return (index % max(1, round(1 / TERNARY_SHARE))) == 0


@dataclass(frozen=True)
class LanguageProfile:
    name: str
    kinds: FrozenSet[str]
    note: str = ""

    def permits(self, kind: str) -> bool:
        return kind in self.kinds

    def filter(self, ops):
        return [o for o in ops if o.kind in self.kinds]


_PREF = {"prefer_rule", "prefer_premise"}

PROFILES: Dict[str, LanguageProfile] = {
    "P_D": LanguageProfile(
        "P_D", frozenset({"premise", "defeasible"} | _PREF),
        "premises and defeasible rules; all three attack forms available"),
    "P_S": LanguageProfile(
        "P_S", frozenset({"premise", "strict", "prefer_premise"}),
        "premises and strict rules; ONLY undermine works, and only premise preferences resolve"),
    "P_S_D": LanguageProfile(
        "P_S_D", frozenset({"premise", "strict", "defeasible"} | _PREF),
        "no axioms, so nothing is unassailable at the root"),
    "FULL": LanguageProfile(
        "FULL", frozenset({"premise", "axiom", "defeasible", "strict"} | _PREF),
        "the full language"),
}

TASK_PROFILES: Dict[str, Tuple[str, ...]] = {
    "status_query": ("P_D", "P_S", "P_S_D", "FULL"),
    "semantics_query": ("P_D", "P_S_D", "FULL"),
    "claim_chain": ("P_D", "P_S_D", "FULL"),
    "defeat_diagnosis": ("P_D", "P_S_D", "FULL"),
    "perturbation": ("P_D", "P_S_D", "FULL"),
    "preference_construction": ("P_D", "P_S_D", "FULL"),
    "counter_argument": ("P_S_D", "FULL"),
    "formalization": ("P_D", "P_S_D", "FULL"),
    "attack_defense": ("P_S_D", "FULL"),
}
