from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Iterable

from arggym.aspic.engine import (
    ASPICFramework,
    Operation,
    contrary,
    DEFAULT_ORDERING,
    JUSTIFIED,
    OVERRULED,
    UNDECIDED,
    UNSATISFIABLE,
    describe_operation,
)
from arggym.aspic.dsl import parse_dsl

VALID_STATUSES = {JUSTIFIED, OVERRULED, UNDECIDED, UNSATISFIABLE}

@dataclass
class Verdict:
    ordering: str
    parse_ok: bool
    operations_applied: int
    dropped: List[dict]
    justified: List[str]
    status_map: Dict[str, str]
    n_arguments: int
    kb: dict
    query: Optional[str] = None
    query_status: Optional[str] = None
    query_negation_status: Optional[str] = None
    consistent: Optional[bool] = None
    invariants_ok: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class InvariantReport:
    ok: bool
    checks: List[dict] = field(default_factory=list)

    def failures(self) -> List[dict]:
        return [c for c in self.checks if not c["ok"]]

    def to_dict(self) -> dict:
        return asdict(self)

class ASPICVerifier:

    def __init__(self, framework: ASPICFramework, dropped: Optional[List[dict]] = None,
                 operations_applied: int = 0):
        self.fw = framework
        self.dropped = dropped or []
        self.operations_applied = operations_applied

    @classmethod
    def from_dsl(cls, text: str, ordering: str = DEFAULT_ORDERING) -> "ASPICVerifier":
        parsed = parse_dsl(text)
        fw = ASPICFramework(ordering=ordering)
        applied = 0
        for op in parsed.operations:
            try:
                fw.apply(op)
                applied += 1
            except ValueError as e:
                parsed.dropped.append({"raw": describe_operation(op), "reason": f"rejected:{e}"})
        return cls(fw, dropped=parsed.dropped, operations_applied=applied)

    @classmethod
    def from_operations(cls, ops: Iterable[Operation],
                        ordering: str = DEFAULT_ORDERING) -> "ASPICVerifier":
        fw = ASPICFramework(ordering=ordering)
        ops = list(ops)
        fw.apply_all(ops)
        return cls(fw, dropped=[], operations_applied=len(ops))

    def status(self, claim: str) -> str:
        return self.fw.status(claim.strip())

    def is_justified(self, claim: str) -> bool:
        return self.fw.is_justified(claim.strip())

    def justified(self) -> List[str]:
        return sorted(self.fw.justified_conclusions())

    def status_map(self) -> Dict[str, str]:
        return self.fw.status_map()

    def _ext_conclusions(self, extensions) -> "List[set]":
        return [{str(a.conclusion) for a in ext} for ext in extensions]

    def preferred_conclusions(self) -> "List[set]":
        return self._ext_conclusions(self.fw.preferred_extensions())

    def stable_conclusions(self) -> "List[set]":
        return self._ext_conclusions(self.fw.stable_extensions())

    def acceptance(self, claim: str, semantics: str = "preferred") -> str:
        claim = claim.strip()
        conc = (self.preferred_conclusions() if semantics == "preferred"
                else self.stable_conclusions())
        if not conc:
            return "no_extension"
        n_in = sum(1 for s in conc if claim in s)
        if n_in == 0:
            return "rejected"
        return "skeptical" if n_in == len(conc) else "credulous"

    def is_floating(self, claim: str) -> bool:
        claim = claim.strip()
        if self.status(claim) == JUSTIFIED:
            return False
        return self.acceptance(claim, "preferred") == "skeptical"

    def _arg_record(self, arg, labels) -> dict:
        af = self.fw.af
        defeaters = list(af.get_incoming_defeat_arguments(arg))
        in_defeaters = [d.short_name for d in defeaters if labels.get(d) == "IN"]
        return {
            "id": arg.short_name,
            "conclusion": str(arg.conclusion),
            "label": labels[arg],
            "premises": sorted(str(p) for p in arg.premises),
            "defeasible_rules": sorted(str(r) for r in arg.defeasible_rules),
            "strict_rules": sorted(str(r) for r in arg.strict_rules),
            "attacked_by": sorted(d.short_name for d in defeaters),
            "defeated_by": sorted(in_defeaters),
        }

    def explain(self, claim: str) -> dict:
        claim = claim.strip()
        labels = self.fw.argument_labels()
        records = [self._arg_record(a, labels)
                   for a in self.fw.af.arguments if str(a.conclusion) == claim]
        return {
            "claim": claim,
            "status": self.fw.status(claim),
            "arguments": records,
            "contrary": {"claim": contrary(claim),
                         "status": self.fw.status(contrary(claim))},
        }

    def attack_graph(self) -> dict:
        labels = self.fw.argument_labels()
        args = [{"id": a.short_name, "conclusion": str(a.conclusion), "label": labels[a]}
                for a in self.fw.af.arguments]
        defeats = [{"from": d.from_argument.short_name, "to": d.to_argument.short_name}
                   for d in self.fw.af.defeats]
        grounded = [a.short_name for a, lab in labels.items() if lab == "IN"]
        return {"arguments": args, "defeats": defeats, "grounded": sorted(grounded)}

    def contradictions(self) -> List[str]:
        sm = self.fw.status_map()
        bad = {l for l in sm if sm[l] == JUSTIFIED and sm.get(contrary(l)) == JUSTIFIED}
        return sorted(bad)

    def is_consistent(self) -> bool:
        return not self.contradictions()

    def check_invariants(self) -> InvariantReport:
        af = self.fw.af
        labels = self.fw.argument_labels()
        grounded = {a for a, lab in labels.items() if lab == "IN"}
        sm = self.fw.status_map()
        checks: List[dict] = []

        def record(name, ok, detail=""):
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        from_ext = set(self.fw.justified_conclusions())
        from_map = {l for l, s in sm.items() if s == JUSTIFIED}
        record("justified_paths_agree", from_ext == from_map,
               f"ext\\map={sorted(from_ext - from_map)} map\\ext={sorted(from_map - from_ext)}"
               if from_ext != from_map else "")

        bad = {l: s for l, s in sm.items() if s not in VALID_STATUSES}
        record("statuses_valid", not bad, str(bad) if bad else "")

        in_law = all(
            all(labels[d] == "OUT" for d in af.get_incoming_defeat_arguments(a))
            for a in grounded
        )
        record("labeling_in_law", in_law)

        out_args = [a for a, lab in labels.items() if lab == "OUT"]
        out_law = all(
            any(labels[d] == "IN" for d in af.get_incoming_defeat_arguments(a))
            for a in out_args
        )
        record("labeling_out_law", out_law)

        undec_args = [a for a, lab in labels.items() if lab == "UNDEC"]
        undec_law = all(
            (not any(labels[d] == "IN" for d in af.get_incoming_defeat_arguments(a)))
            and any(labels[d] != "OUT" for d in af.get_incoming_defeat_arguments(a))
            for a in undec_args
        )
        record("labeling_undec_law", undec_law)

        conflict_free = all(
            not (set(af.get_incoming_defeat_arguments(a)) & grounded)
            for a in grounded
        )
        record("grounded_conflict_free", conflict_free)

        return InvariantReport(ok=all(c["ok"] for c in checks), checks=checks)

    def verdict(self, query: Optional[str] = None, check: bool = True) -> Verdict:
        sm = self.fw.status_map()
        v = Verdict(
            ordering=self.fw._ordering_name,
            parse_ok=(len(self.dropped) == 0),
            operations_applied=self.operations_applied,
            dropped=self.dropped,
            justified=sorted(self.fw.justified_conclusions()),
            status_map=sm,
            n_arguments=len(self.fw.arguments),
            kb=self.fw.get_kb_snapshot(),
        )
        if query is not None:
            q = query.strip()
            v.query = q
            v.query_status = self.fw.status(q)
            v.query_negation_status = self.fw.status(contrary(q))
        if check:
            v.consistent = self.is_consistent()
            v.invariants_ok = self.check_invariants().ok
        return v

def verify(dsl_text: str, query: Optional[str] = None,
           ordering: str = DEFAULT_ORDERING, check: bool = True) -> Verdict:
    return ASPICVerifier.from_dsl(dsl_text, ordering=ordering).verdict(query=query, check=check)
