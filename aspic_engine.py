from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
from py_arg.aspic_classes.literal import Literal
from py_arg.aspic_classes.defeasible_rule import DefeasibleRule
from py_arg.aspic_classes.strict_rule import StrictRule
from py_arg.aspic_classes.argumentation_system import ArgumentationSystem
from py_arg.aspic_classes.argumentation_theory import ArgumentationTheory
from py_arg.aspic_classes.orderings.preference_preorder import PreferencePreorder
from py_arg.aspic_classes.orderings.argument_orderings.last_link_ordering import (
    LastLinkElitistOrdering,
    LastLinkDemocraticOrdering,
)
from py_arg.aspic_classes.orderings.argument_orderings.weakest_link_ordering import (
    WeakestLinkElitistOrdering,
    WeakestLinkDemocraticOrdering,
)
from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension

ORDERINGS = {
    "last_link_elitist": LastLinkElitistOrdering,
    "last_link_democratic": LastLinkDemocraticOrdering,
    "weakest_link_elitist": WeakestLinkElitistOrdering,
    "weakest_link_democratic": WeakestLinkDemocraticOrdering,
}
DEFAULT_ORDERING = "last_link_elitist"

JUSTIFIED = "JUSTIFIED"
OVERRULED = "OVERRULED"
UNDECIDED = "UNDECIDED"
UNSATISFIABLE = "UNSATISFIABLE"


def contrary(s: str) -> str:
    return s[1:] if s.startswith("-") else "-" + s


def describe_operation(op: "Operation") -> str:
    if op.kind in ("premise", "axiom"):
        return f"[{op.kind}: {op.content}]"
    if op.kind == "defeasible":
        return f"[defeasible: {' AND '.join(op.antecedents)} => {op.consequent}]"
    if op.kind == "strict":
        return f"[strict: {' AND '.join(op.antecedents)} -> {op.consequent}]"
    if op.kind in ("prefer_rule", "prefer_premise"):
        return f"[{op.kind}: {op.stronger} > {op.weaker}]"
    return f"[{op.kind}]"

@dataclass(frozen=True)
class Operation:
    kind: str
    content: Optional[str] = None          
    name: Optional[str] = None             
    antecedents: Tuple[str, ...] = ()      
    consequent: Optional[str] = None      
    stronger: Optional[str] = None        
    weaker: Optional[str] = None         

class ASPICFramework:
    def __init__(self, ordering: str = DEFAULT_ORDERING):
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {sorted(ORDERINGS)}, got {ordering!r}")
        self._ordering_name = ordering

        self._axioms: List[str] = []
        self._ordinary: List[str] = []
        self._strict: List[Tuple[str, Tuple[str, ...], str]] = []
        self._defeasible: List[Tuple[str, Tuple[str, ...], str]] = []
        self._rule_prefs: List[Tuple[str, str]] = []
        self._premise_prefs: List[Tuple[str, str]] = []

        self._literals: Dict[str, Literal] = {}
        self._contraries: Dict[str, Set[str]] = {}
        self._rule_names: Set[str] = set()

        self._theory: Optional[ArgumentationTheory] = None
        self._af = None
        self._dirty = True


    def _lit(self, s: str) -> Literal:
        if s not in self._literals:
            self._literals[s] = Literal(s)
        return self._literals[s]

    def _register(self, s: str) -> None:
        pos = s[1:] if s.startswith("-") else s
        neg = "-" + pos
        self._lit(pos)
        self._lit(neg)
        self._contraries.setdefault(pos, set()).add(neg)
        self._contraries.setdefault(neg, set()).add(pos)

    def add_axiom(self, content: str) -> None:
        self._dirty = True
        self._register(content)
        self._axioms.append(content)

    def add_ordinary_premise(self, content: str) -> None:
        self._dirty = True
        self._register(content)
        self._ordinary.append(content)

    def add_premise(self, content: str, axiom: bool = False) -> None:
        (self.add_axiom if axiom else self.add_ordinary_premise)(content)

    def add_strict_rule(self, name: str, antecedents: Iterable[str], consequent: str) -> None:
        self._add_rule(self._strict, name, antecedents, consequent)

    def add_defeasible_rule(self, name: str, antecedents: Iterable[str], consequent: str) -> None:
        self._add_rule(self._defeasible, name, antecedents, consequent)

    def _add_rule(self, bucket, name, antecedents, consequent) -> None:
        self._dirty = True
        ants = tuple(antecedents)
        for a in ants:
            self._register(a)
        self._register(consequent)
        bucket.append((name, ants, consequent))
        self._rule_names.add(name)

    def add_rule_preference(self, stronger: str, weaker: str) -> None:
        names = {r[0] for r in self._defeasible}
        if stronger not in names or weaker not in names:
            raise ValueError(f"rule preference references unknown defeasible rule(s): {stronger!r} > {weaker!r}")
        if stronger == weaker:
            raise ValueError("a rule cannot be preferred to itself")
        self._dirty = True
        if (stronger, weaker) not in self._rule_prefs:
            self._rule_prefs.append((stronger, weaker))

    def add_premise_preference(self, stronger: str, weaker: str) -> None:
        if stronger not in self._ordinary or weaker not in self._ordinary:
            raise ValueError(f"premise preference references unknown ordinary premise(s): {stronger!r} > {weaker!r}")
        if stronger == weaker:
            raise ValueError("a premise cannot be preferred to itself")
        self._dirty = True
        if (stronger, weaker) not in self._premise_prefs:
            self._premise_prefs.append((stronger, weaker))

    def apply(self, op: Operation) -> None:
        if op.kind == "axiom":
            self.add_axiom(op.content)
        elif op.kind == "premise":
            self.add_ordinary_premise(op.content)
        elif op.kind == "strict":
            self.add_strict_rule(op.name, op.antecedents, op.consequent)
        elif op.kind == "defeasible":
            self.add_defeasible_rule(op.name, op.antecedents, op.consequent)
        elif op.kind == "prefer_rule":
            self.add_rule_preference(op.stronger, op.weaker)
        elif op.kind == "prefer_premise":
            self.add_premise_preference(op.stronger, op.weaker)
        else:
            raise ValueError(f"unknown operation kind: {op.kind!r}")

    def apply_all(self, ops: Iterable[Operation]) -> None:
        for op in ops:
            self.apply(op)

    @staticmethod
    def _closure(pairs: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        closure = set(pairs)
        changed = True
        while changed:
            changed = False
            for a, b in list(closure):
                for c, d in list(closure):
                    if b == c and (a, d) not in closure:
                        closure.add((a, d))
                        changed = True
        return closure

    def build(self) -> None:
        if not self._dirty and self._theory is not None:
            return

        language = dict(self._literals)
        contraries = {
            s: {language[c] for c in cs if c in language}
            for s, cs in self._contraries.items()
            if s in language
        }

        strict = [
            StrictRule(n, {language[a] for a in ants}, language[c])
            for n, ants, c in self._strict
        ]
        defeasible: List[DefeasibleRule] = []
        rule_by_name: Dict[str, DefeasibleRule] = {}
        for n, ants, c in self._defeasible:
            d = DefeasibleRule(n, {language[a] for a in ants}, language[c])
            defeasible.append(d)
            rule_by_name[n] = d

        rule_pairs = [(d, d) for d in defeasible]
        for stronger, weaker in self._closure(self._rule_prefs):
            if stronger in rule_by_name and weaker in rule_by_name:
                rule_pairs.append((rule_by_name[weaker], rule_by_name[stronger]))

        arg_sys = ArgumentationSystem(
            language=language,
            contraries_and_contradictories=contraries,
            strict_rules=strict,
            defeasible_rules=defeasible,
            defeasible_rule_preferences=PreferencePreorder(rule_pairs),
        )

        ordinary_lits = [language[o] for o in self._ordinary]
        premise_pairs = [(p, p) for p in ordinary_lits]
        for stronger, weaker in self._closure(self._premise_prefs):
            if stronger in language and weaker in language:
                premise_pairs.append((language[weaker], language[stronger]))

        self._theory = ArgumentationTheory(
            argumentation_system=arg_sys,
            knowledge_base_axioms=[language[a] for a in self._axioms],
            knowledge_base_ordinary_premises=ordinary_lits,
            ordinary_premise_preferences=PreferencePreorder(premise_pairs),
        )

        ordering = ORDERINGS[self._ordering_name](
            arg_sys.rule_preferences, self._theory.ordinary_premise_preferences
        )
        self._af = self._theory.create_abstract_argumentation_framework("fw", ordering=ordering)
        self._dirty = False

    @property
    def theory(self) -> ArgumentationTheory:
        self.build()
        return self._theory

    @property
    def af(self):
        self.build()
        return self._af

    @property
    def arguments(self):
        return list(self.theory.all_arguments)

    def grounded_extension(self) -> Set:
        return set(get_grounded_extension(self.af))
    
    def argument_labels(self) -> Dict[object, str]:
        af = self.af
        grounded = set(get_grounded_extension(af))
        labels: Dict[object, str] = {}
        for arg in af.arguments:
            if arg in grounded:
                labels[arg] = "IN"
            else:
                defeaters = set(af.get_incoming_defeat_arguments(arg))
                labels[arg] = "OUT" if (defeaters & grounded) else "UNDEC"
        return labels

    _argument_labels = argument_labels

    def status_map(self) -> Dict[str, str]:
        labels = self.argument_labels()
        by_conclusion: Dict[str, List[str]] = defaultdict(list)
        for arg, lab in labels.items():
            by_conclusion[str(arg.conclusion)].append(lab)

        out: Dict[str, str] = {}
        for lit, labs in by_conclusion.items():
            labset = set(labs)
            if "IN" in labset:
                out[lit] = JUSTIFIED
            elif "UNDEC" in labset:
                out[lit] = UNDECIDED
            else:
                out[lit] = OVERRULED
        return out

    def status(self, claim: str) -> str:
        return self.status_map().get(claim, UNSATISFIABLE)

    def is_justified(self, claim: str) -> bool:
        return self.status(claim) == JUSTIFIED

    def justified_conclusions(self) -> Set[str]:
        return {str(a.conclusion) for a in self.grounded_extension()}

    def get_kb_snapshot(self) -> Dict:
        return {
            "axioms": list(self._axioms),
            "ordinary_premises": list(self._ordinary),
            "strict_rules": [[n, list(a), c] for n, a, c in self._strict],
            "defeasible_rules": [[n, list(a), c] for n, a, c in self._defeasible],
            "rule_preferences": [[s, w] for s, w in self._rule_prefs],
            "premise_preferences": [[s, w] for s, w in self._premise_prefs],
        }

    def clone(self) -> "ASPICFramework":
        fw = ASPICFramework(ordering=self._ordering_name)
        fw._axioms = list(self._axioms)
        fw._ordinary = list(self._ordinary)
        fw._strict = list(self._strict)
        fw._defeasible = list(self._defeasible)
        fw._rule_prefs = list(self._rule_prefs)
        fw._premise_prefs = list(self._premise_prefs)
        fw._literals = dict(self._literals)
        fw._contraries = {k: set(v) for k, v in self._contraries.items()}
        fw._rule_names = set(self._rule_names)
        fw._dirty = True
        return fw

    def is_rule_name(self, name: str) -> bool:
        return name in self._rule_names
