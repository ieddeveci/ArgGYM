from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple


ANCHOR_MAX = 5         
MAX_LEVEL = 15       


FEATURES: Dict[str, int] = {
    "conflict": 2,          
    "negation": 2,          
    "axioms": 3,            
    "side_branches": 3,    
    "derived_attacks": 3,  
    "strict_rules": 4,    
    "attack_chains": 4,    
    "undercuts": 5,       
}


def has(feature: str, level: int) -> bool:
    return level >= FEATURES[feature]


@dataclass(frozen=True)
class Knob:
    anchors: Tuple[float, ...]
    slope: float = 0.0
    lo: float = 0.0
    hi: float = float("inf")

    def at(self, level: int) -> int:
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")
        if level <= ANCHOR_MAX:
            v = self.anchors[level - 1]
        else:
            v = self.anchors[ANCHOR_MAX - 1] + self.slope * (level - ANCHOR_MAX)
        return int(min(max(v, self.lo), self.hi))


@dataclass(frozen=True)
class Gate:
    anchors: Tuple[bool, ...]

    def at(self, level: int) -> bool:
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")
        return self.anchors[min(level, ANCHOR_MAX) - 1]


def _ge(threshold: int) -> Gate:
    return Gate(tuple(lv >= threshold for lv in range(1, ANCHOR_MAX + 1)))


def _eq1() -> Gate:
    return Gate((True, False, False, False, False))


class Recipe:

    def __init__(self, table: dict):
        self._table = dict(table)
        self._max = max(self._table)

    def at(self, level: int):
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")
        return self._table[min(level, self._max)]

_THEORY_ANCHORS: Dict[int, dict] = {
    1: dict(n_atoms=3, n_premises=2, n_axioms=0, n_rules=1, n_strict=0, n_pref=0,
            p_conflict=0.0, p_undercut=0.0, max_arity=1, layers=1,
            require_conflict=False, min_arguments=1, max_tries=200),
    2: dict(n_atoms=4, n_premises=2, n_axioms=0, n_rules=3, n_strict=0, n_pref=0,
            p_conflict=0.5, p_undercut=0.0, max_arity=2, layers=2,
            require_conflict=True, min_arguments=1, max_tries=200),
    3: dict(n_atoms=6, n_premises=3, n_axioms=1, n_rules=4, n_strict=0, n_pref=0,
            p_conflict=0.55, p_undercut=0.0, max_arity=2, layers=3,
            require_conflict=True, min_arguments=2, max_tries=200),
    4: dict(n_atoms=7, n_premises=3, n_axioms=1, n_rules=6, n_strict=1, n_pref=1,
            p_conflict=0.6, p_undercut=0.0, max_arity=2, layers=3,
            require_conflict=True, min_arguments=2, max_tries=220),
    5: dict(n_atoms=9, n_premises=4, n_axioms=1, n_rules=8, n_strict=2, n_pref=2,
            p_conflict=0.65, p_undercut=0.30, max_arity=3, layers=4,
            require_conflict=True, min_arguments=2, max_tries=240),
}


def _theory_kwargs(level: int) -> dict:
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    if level <= ANCHOR_MAX:
        return dict(_THEORY_ANCHORS[level])
    d = level - ANCHOR_MAX
    base = _THEORY_ANCHORS[ANCHOR_MAX]
    return dict(
        n_atoms=base["n_atoms"] + 2 * d,
        n_premises=min(base["n_premises"] + (d + 1) // 2, 10),
        n_axioms=min(base["n_axioms"] + d // 3, 3),
        n_rules=base["n_rules"] + 2 * d,
        n_strict=min(base["n_strict"] + (d + 1) // 2, 6),
        n_pref=min(base["n_pref"] + d, 8),
        p_conflict=min(base["p_conflict"] + 0.02 * d, 0.80),
        p_undercut=min(base["p_undercut"] + 0.05 * d, 0.60),
        max_arity=3,
        layers=min(base["layers"] + (d + 1) // 2, 8),
        require_conflict=True,
        min_arguments=2,
        max_tries=base["max_tries"] + 40 * d,
    )


class _TheorySchedule(dict):

    def __init__(self, config_cls):
        super().__init__()
        self._cls = config_cls
        for lv in range(1, ANCHOR_MAX + 1):
            self[lv] = config_cls(**_theory_kwargs(lv))

    def __missing__(self, level):
        cfg = self._cls(**_theory_kwargs(level))
        self[level] = cfg
        return cfg


def make_theory_schedule(config_cls) -> _TheorySchedule:
    return _TheorySchedule(config_cls)

TASK_KNOBS: Dict[str, Dict[str, Knob]] = {
    "kb_theory": {
        "prem":  Knob((1, 2, 3, 3, 4), slope=0.5, lo=1, hi=8),
        "rules": Knob((1, 3, 4, 6, 8), slope=2.0, lo=1, hi=24),
        "prefs": Knob((0, 0, 0, 1, 2), slope=1.0, hi=8),
    },
    "status_query": {
        "prem_min":  Knob((1, 2, 2, 3, 4), slope=0.5, lo=1, hi=8),
        "prem_max":  Knob((3, 3, 4, 4, 5), slope=0.5, lo=1, hi=10),
        "rule_min":  Knob((1, 2, 3, 3, 4), slope=1.2, lo=1, hi=18),
        "rule_max":  Knob((2, 4, 4, 6, 6), slope=1.4, lo=1, hi=24),
        "n_axioms":  Knob((0, 1, 1, 1, 1), slope=0.25, hi=3),
        "n_strict":  Knob((0, 0, 1, 1, 2), slope=0.5, hi=5),
        "n_pref":    Knob((0, 0, 1, 1, 2), slope=1.0, hi=8),
        "n_claims":  Knob((2, 4, 6, 7, 8), slope=1.2, lo=1, hi=14),  
    },
    "attack": {
        "depth": Knob((2, 2, 3, 3, 4), slope=0.5, lo=2, hi=8),
    },
    "attackers_of": {
        "depth":       Knob((1, 1, 2, 2, 3), slope=0.5, lo=1, hi=6),
        "n_attackers": Knob((1, 1, 2, 2, 3), slope=0.5, lo=1, hi=6),
    },
    "claim_identification": {
        "n_chains": Knob((1, 1, 2, 2, 2), slope=0.5, lo=1, hi=7),
        "depth":    Knob((2, 3, 2, 3, 3), slope=0.5, lo=2, hi=9),
    },
    "formalization": {
        "n_elements": Knob((2, 4, 6, 8, 10), slope=2.0, lo=2, hi=24),
        "prem":   Knob((1, 1, 1, 2, 2), slope=0.5, lo=1, hi=5),
        "ax":     Knob((0, 1, 1, 1, 2), slope=0.25, hi=4),
        "rules":  Knob((1, 2, 4, 5, 6), slope=2.0, lo=1, hi=16),
        "strict": Knob((0, 0, 0, 1, 2), slope=0.5, hi=5),
        "arity":  Knob((1, 2, 2, 2, 2), slope=0.0, lo=1, hi=3),
    },
    "enthymeme": {
        "n_missing": Knob((1, 2, 3, 4, 5), slope=1.0, lo=1, hi=8),
    },
    "evidence_construction": {
        "n_distract": Knob((1, 2, 3, 4, 5), slope=1.0, hi=10),
        "n_oppose":   Knob((0, 0, 1, 1, 2), slope=0.5, hi=5),
        "min_prem":   Knob((1, 2, 2, 3, 3), slope=0.25, lo=1, hi=6),
    },
    "perturbation_prediction": {
        "n_queries": Knob((1, 2, 3, 4, 5), slope=1.0, lo=1, hi=15),
        "n_perturb": Knob((1, 1, 2, 2, 3), slope=0.7, lo=1, hi=10),
    },
    "preference_construction": {
        "depth_min": Knob((1, 1, 1, 2, 2), slope=0.5, lo=1, hi=5),
        "depth_max": Knob((2, 2, 2, 3, 3), slope=0.5, lo=1, hi=6),
        "con_min":   Knob((1, 1, 1, 2, 2), slope=0.5, lo=1, hi=5),
        "con_max":   Knob((1, 1, 2, 2, 3), slope=0.5, lo=1, hi=6),
        "decoy_min": Knob((0, 0, 0, 1, 1), slope=0.25, hi=3),
        "decoy_max": Knob((0, 1, 1, 2, 2), slope=0.5, hi=4),
        "f_axiom":   Knob((0, 1, 1, 1, 1), slope=0.0, hi=1),
        "f_strict":  Knob((0, 0, 1, 1, 1), slope=0.0, hi=1),
        "f_undercut": Knob((0, 0, 0, 0, 1), slope=0.0, hi=1),
    },
}

TASK_GATES: Dict[str, Dict[str, Gate]] = {
    "kb_theory": {
        "prem":  Knob((1, 2, 3, 3, 4), slope=0.5, lo=1, hi=8),
        "rules": Knob((1, 3, 4, 6, 8), slope=2.0, lo=1, hi=24),
        "prefs": Knob((0, 0, 0, 1, 2), slope=1.0, hi=8),
    },
    "status_query": {
        "easy_no_conflict": _eq1(),           
        "require_conflict": _ge(2),
        "undercuts":        _ge(FEATURES["undercuts"]),
        "kb_disclaim_required": Gate((False, True, True, True, True)),
        "easy_single_stance": _eq1(),
        "kb_prefs":         _ge(3),
        "kb_perturb":       _ge(3),
    },
    "kb_theory": {
        "easy_single":  _eq1(),
        "allow_ax":     _ge(FEATURES["axioms"]),
        "allow_strict": _ge(FEATURES["strict_rules"]),
        "undercuts":    _ge(FEATURES["undercuts"]),
    },
    "attack": {
        "axiom_distractor": _ge(2),           
        "strict_on_path":   _ge(3),
        "side_branch":      _ge(FEATURES["side_branches"]),
        "allow_ax":         _ge(FEATURES["axioms"]),
        "allow_strict":     _ge(FEATURES["strict_rules"]),
    },
    "attackers_of": {
        "derived_attacker": _ge(FEATURES["derived_attacks"]),
        "second_decoy":     _ge(4),
        "allow_ax":         _ge(FEATURES["axioms"]),
        "allow_strict":     _ge(FEATURES["strict_rules"]),
    },
    "claim_identification": {
        "allow_neg":    _ge(FEATURES["negation"]),
        "allow_ax":     _ge(FEATURES["axioms"]),
        "allow_strict": _ge(FEATURES["strict_rules"]),
        "defeater":     _ge(5),            
    },
    "formalization": {
        "neg": Gate((False, True, True, True, True)),
    },
    "robustness": {
        "simple_merge": _eq1(),
    },
}

TASK_RECIPES: Dict[str, Recipe] = {
    "attackers_of_pool": Recipe({
        1: ["rebut"],
        2: ["rebut", "undermine"],
        3: ["rebut", "undermine"],
        4: ["rebut", "undermine"],
        5: ["rebut", "undermine", "undercut"],
    }),
    "perturbation_kinds": Recipe({
        1: ["retract_premise", "add_premise"],
        2: ["retract_premise", "add_premise", "add_rule"],
        3: ["retract_premise", "add_premise", "add_rule", "remove_rule", "add_pref"],
        4: ["retract_premise", "add_premise", "add_rule", "remove_rule", "add_pref",
            "undercut", "downgrade_axiom"],
        5: ["retract_premise", "add_premise", "add_rule", "remove_rule", "add_pref",
            "undercut", "downgrade_axiom"],
    }),
    "robustness_variant": Recipe({
        1: ("single", "defeat"), 2: ("single", "defeat"), 3: ("set", "defeat"),
        4: ("single", "reinstate"), 5: ("pair", "defeat"),
    }),
    "preference_templates": Recipe({
        1: ["direct", "intermediate_target"],
        2: ["direct", "intermediate_target", "upstream"],
        3: ["direct", "intermediate_target", "upstream", "two_supports"],
        4: ["direct", "upstream", "intermediate_target", "two_supports", "reinstatement"],
        5: ["direct", "upstream", "two_supports", "reinstatement", "prefer_premise"],
    }),
}


def knob(task: str, name: str, level: int) -> int:
    return TASK_KNOBS[task][name].at(level)


def gate(task: str, name: str, level: int) -> bool:
    return TASK_GATES[task][name].at(level)


def recipe(name: str, level: int):
    return TASK_RECIPES[name].at(level)


def task_elements(kind: str, level: int) -> int:
    if kind == "formalization":
        return knob("formalization", "n_elements", level)
    if kind == "enthymeme":
        return knob("enthymeme", "n_missing", level)
    return 0


def content_band(level: int, n_bands: int = ANCHOR_MAX) -> int:
    return min(level, n_bands) - 1
