"""Operations to JSON and back.

A frozen row has to carry the theory it was built from, or scoring it needs the
generator that produced it. The v1 harness shows what the absence costs: to
score a stored generation it calls `_rebuild(task, level, ordering, seed)` and
regenerates the item, which pins every stored run to one generator commit.

`theory_schema` is versioned separately from the package. `Operation` is a
dataclass, so adding a field would make old rows deserialize with a default and
quietly mean something new; `from_dict` refuses a schema it does not know
instead.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from arggym.aspic.engine import Operation

THEORY_SCHEMA = 1

# The fields that carry meaning per kind. Writing only these keeps a row
# readable and keeps a diff between two tasksets to the parts that differ.
_BY_KIND = {
    "premise": ("content",),
    "axiom": ("content",),
    "defeasible": ("name", "antecedents", "consequent"),
    "strict": ("name", "antecedents", "consequent"),
    "prefer_rule": ("stronger", "weaker"),
    "prefer_premise": ("stronger", "weaker"),
}


def op_to_dict(op: Operation) -> Dict[str, Any]:
    if op.kind not in _BY_KIND:
        raise ValueError(f"cannot serialize operation of kind {op.kind!r}")
    out: Dict[str, Any] = {"kind": op.kind}
    for f in _BY_KIND[op.kind]:
        v = getattr(op, f)
        out[f] = list(v) if f == "antecedents" else v
    return out


def op_from_dict(d: Dict[str, Any]) -> Operation:
    kind = d.get("kind")
    if kind not in _BY_KIND:
        raise ValueError(f"unknown operation kind {kind!r}")
    kw: Dict[str, Any] = {"kind": kind}
    for f in _BY_KIND[kind]:
        v = d.get(f)
        # Operation is frozen and antecedents is a Tuple; JSON gives a list, and
        # a list here would make two equal operations compare unequal rather
        # than raise, so the round trip has to put the tuple back.
        kw[f] = tuple(v or ()) if f == "antecedents" else v
    return Operation(**kw)


def ops_to_json(ops: Iterable[Operation]) -> List[Dict[str, Any]]:
    return [op_to_dict(o) for o in ops]


def ops_from_json(rows: Iterable[Dict[str, Any]]) -> List[Operation]:
    return [op_from_dict(r) for r in rows]


def check_schema(version: Any) -> None:
    """Refuse a theory this build cannot read, rather than misread it."""
    if version != THEORY_SCHEMA:
        raise ValueError(
            f"theory_schema {version!r} was written by a different ArgGYM; "
            f"this build reads schema {THEORY_SCHEMA}")
