from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from arggym.aspic import aspic
except ImportError:
    import aspic

Theory = aspic.Theory
GROUNDED, PREFERRED, STABLE = aspic.GROUNDED, aspic.PREFERRED, aspic.STABLE
SEMISTABLE, IDEAL, EAGER, COMPLETE = (aspic.SEMISTABLE, aspic.IDEAL,
                                      aspic.EAGER, aspic.COMPLETE)
LAST_LINK_ELITIST = aspic.LAST_LINK_ELITIST
LAST_LINK_DEMOCRATIC = aspic.LAST_LINK_DEMOCRATIC
WEAKEST_LINK_ELITIST = aspic.WEAKEST_LINK_ELITIST
WEAKEST_LINK_DEMOCRATIC = aspic.WEAKEST_LINK_DEMOCRATIC

ORDERINGS = {
    "last_link_elitist": LAST_LINK_ELITIST,
    "last_link_democratic": LAST_LINK_DEMOCRATIC,
    "weakest_link_elitist": WEAKEST_LINK_ELITIST,
    "weakest_link_democratic": WEAKEST_LINK_DEMOCRATIC,
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
        self._t = Theory(ORDERINGS[ordering])
        self._axioms: List[str] = []
        self._ordinary: List[str] = []
        self._strict: List[Tuple[str, Tuple[str, ...], str]] = []
        self._defeasible: List[Tuple[str, Tuple[str, ...], str]] = []
        self._rule_prefs: List[Tuple[str, str]] = []
        self._premise_prefs: List[Tuple[str, str]] = []
        self._rule_names: Set[str] = set()


    def add_axiom(self, content: str) -> None:
        self._axioms.append(content)
        self._t.axiom(content)

    def add_ordinary_premise(self, content: str) -> None:
        self._ordinary.append(content)
        self._t.premise(content)

    def add_premise(self, content: str, axiom: bool = False) -> None:
        (self.add_axiom if axiom else self.add_ordinary_premise)(content)

    def add_strict_rule(self, name: str, antecedents: Iterable[str], consequent: str) -> None:
        ants = tuple(antecedents)
        self._strict.append((name, ants, consequent))
        self._rule_names.add(name)
        self._t.strict(name, list(ants), consequent)

    def add_defeasible_rule(self, name: str, antecedents: Iterable[str], consequent: str) -> None:
        ants = tuple(antecedents)
        self._defeasible.append((name, ants, consequent))
        self._rule_names.add(name)
        self._t.defeasible(name, list(ants), consequent)

    def add_rule_preference(self, stronger: str, weaker: str) -> None:
        names = {r[0] for r in self._defeasible}
        if stronger not in names or weaker not in names:
            raise ValueError(
                f"rule preference references unknown defeasible rule(s): {stronger!r} > {weaker!r}")
        if stronger == weaker:
            raise ValueError("a rule cannot be preferred to itself")
        if (stronger, weaker) not in self._rule_prefs:
            self._rule_prefs.append((stronger, weaker))
            self._t.prefer_rule(stronger, weaker)

    def add_premise_preference(self, stronger: str, weaker: str) -> None:
        if stronger not in self._ordinary or weaker not in self._ordinary:
            raise ValueError(
                f"premise preference references unknown ordinary premise(s): "
                f"{stronger!r} > {weaker!r}")
        if stronger == weaker:
            raise ValueError("a premise cannot be preferred to itself")
        if (stronger, weaker) not in self._premise_prefs:
            self._premise_prefs.append((stronger, weaker))
            self._t.prefer_premise(stronger, weaker)

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

    def build(self) -> None:
        self._t.framework()


    @property
    def theory(self):
        return self._t._build()

    @property
    def af(self):
        return self._t.framework()

    @property
    def arguments(self):
        return list(self._t.framework().arguments)


    def grounded_extension(self) -> Set:
        return set(self._t.extensions_as_arguments(GROUNDED)[0]) \
            if self._t.extensions_as_arguments(GROUNDED) else set()


    class _Concl:
        __slots__ = ("conclusion",)

        def __init__(self, c):
            self.conclusion = c

        def __repr__(self):
            return f"<{self.conclusion}>"

        def __hash__(self):
            return hash(self.conclusion)

        def __eq__(self, other):
            return getattr(other, "conclusion", None) == self.conclusion

    def _fast(self, semantics) -> "List[Set]":
        return [{ASPICFramework._Concl(c) for c in ext}
                for ext in self._t.extensions(semantics)]

    def preferred_extensions(self) -> "List[Set]":
        return self._fast(PREFERRED)

    def stable_extensions(self) -> "List[Set]":
        return self._fast(STABLE)

    def semistable_extensions(self) -> "List[Set]":
        return self._fast(SEMISTABLE)

    def complete_extensions(self) -> "List[Set]":
        return self._fast(COMPLETE)

    def preferred_extension_arguments(self) -> "List[Set]":
        return [set(e) for e in self._t.extensions_as_arguments(PREFERRED)]

    def ideal_extension(self) -> "Set":
        e = self._t.extensions(IDEAL)
        return {ASPICFramework._Concl(c) for c in e[0]} if e else set()

    def eager_extension(self) -> "Set":
        e = self._t.extensions(EAGER)
        return {ASPICFramework._Concl(c) for c in e[0]} if e else set()

    def argument_labels(self) -> Dict[object, str]:
        lab = self._t.labelling(GROUNDED)
        if lab is None:
            return {a: "UNDEC" for a in self._t.framework().arguments}
        m = {aspic.IN: "IN", aspic.OUT: "OUT", aspic.UNDEC: "UNDEC"}
        return {a: m[lab.label_of(a)] for a in self._t.framework().arguments}

    _argument_labels = argument_labels

    def status_map(self) -> Dict[str, str]:
        labels = self.argument_labels()
        by_conclusion: Dict[str, List[str]] = defaultdict(list)
        for arg, lab in labels.items():
            by_conclusion[str(arg.conclusion)].append(lab)
        out: Dict[str, str] = {}
        for lit in sorted(by_conclusion):
            labset = set(by_conclusion[lit])
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
        return set(self._t.extensions(GROUNDED)[0]) if self._t.extensions(GROUNDED) else set()


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
        fw._rule_names = set(self._rule_names)
        fw._t = self._t.copy()
        return fw

    def is_rule_name(self, name: str) -> bool:
        return name in self._rule_names


    def validate(self) -> List[str]:
        return self._t.validate()

    def justification_status(self, claim: str, semantics: str = "preferred") -> Set[str]:
        sem = {"preferred": PREFERRED, "stable": STABLE, "grounded": GROUNDED,
               "semistable": SEMISTABLE, "complete": COMPLETE}[semantics]
        return self._t.justification_status(claim, sem)
