"""A row, back into the objects its scorer reads.

`core/serialize.py` round-trips operations. This module round-trips everything
else a scorer touches, so a line of a frozen JSONL scores to the same number as
the item that produced it and the generator is not needed at scoring time. The
v1 harness shows the cost of the gap: to score a stored generation it called
`_rebuild(task, level, ordering, seed)` and regenerated the item, which pinned
every stored run to one generator commit.

Three rules shape what a row carries.

**Everything answer-bearing sits under `metadata.gold`**, so "do not show the
model `metadata.gold`" is one rule rather than twelve. `min_directives` is there
because the prompt never states how many directives are needed, and
`perturbation.survivors` because it names the claims that did *not* change.
Everything else a scorer reads and the question already gives away -- the
queried literals, the goals, the claim -- sits under `metadata.state`.

**Nothing derived is written.** `attack_defense`'s `subgoals` is a property over
the goals and the theory, so the row records that the task uses it and the
scoring path recomputes it. A serialized copy would drift the first time the
property changed.

**A missing key refuses.** Defaulting would turn a row that lost a field into a
row that scores differently, which is the failure this module exists to prevent,
so every lookup raises `MissingField` naming the task and the key.

Which fields a task carries is the table in `core/registry.py`, not a branch
here. `core/export.py:39` still holds the if/elif ladder this replaces, whose
default failure is a forgotten branch -- that is how two tasks went missing from
the export (#23).
"""
from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from arggym.aspic.engine import Operation
from arggym.core import registry
from arggym.core.answers import ScoreResult
from arggym.core.serialize import check_schema, ops_from_json, ops_to_json


class EngineMismatch(RuntimeError):
    """A row was built against a different engine than this one."""


class MissingField(KeyError):
    """A row does not carry something its scorer reads.

    A `KeyError` so a harness that already guards row access keeps working, with
    the message in the message rather than in a repr of the key.
    """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0] if self.args else ""


# -- codecs ------------------------------------------------------------------
# Each is (encode, decode) over one field, given the theory operations for the
# fields that are expressed relative to them.

def _plain(v: Any, base: Sequence[Operation]) -> Any:
    return v


def _enc_ops(v: Sequence[Operation], base: Sequence[Operation]) -> Any:
    return ops_to_json(v)


def _dec_ops(v: Any, base: Sequence[Operation]) -> List[Operation]:
    return ops_from_json(v)


def _enc_ops_index(v: Sequence[Operation], base: Sequence[Operation]) -> List[int]:
    """A sublist of the theory, written as positions in it.

    `claim_chain.line_ops` is a sublist of `base_ops` by object identity.
    Writing the operations again would let the copy disagree with the theory,
    and would say twice what the row already says once.
    """
    out = []
    for o in v:
        pos = next((i for i, b in enumerate(base) if b is o), None)
        if pos is None:
            raise ValueError(f"operation {o!r} is not part of the theory it indexes")
        out.append(pos)
    return out


def _dec_ops_index(v: Any, base: Sequence[Operation]) -> List[Operation]:
    out = []
    for i in v:
        if not isinstance(i, int) or not 0 <= i < len(base):
            raise ValueError(f"theory index {i!r} is outside a theory of {len(base)}")
        out.append(base[i])
    return out


def _enc_goals(v: Sequence[Dict], base: Sequence[Operation]) -> List[Dict[str, str]]:
    # `current` is the status before the answer, which the prompt states and no
    # scorer reads, so the row keeps the two keys `score_item` looks at.
    return [{"claim": g["claim"], "want": g["want"]} for g in v]


def _dec_goals(v: Any, base: Sequence[Operation]) -> List[Dict[str, str]]:
    return [{"claim": g["claim"], "want": g["want"]} for g in v]


def _enc_pairs(v: Sequence[Tuple[str, str]], base: Sequence[Operation]) -> List[List[str]]:
    return [[a, b] for a, b in v]


def _dec_pairs(v: Any, base: Sequence[Operation]) -> List[Tuple[str, str]]:
    # JSON has no tuples, and `semantics_query` keys its gold by (claim,
    # semantics). A list here would miss every lookup instead of raising.
    return [(a, b) for a, b in v]


def _enc_pair_map(v: Dict[Tuple[str, str], str], base: Sequence[Operation]) -> List[List[str]]:
    return [[a, b, val] for (a, b), val in v.items()]


def _dec_pair_map(v: Any, base: Sequence[Operation]) -> Dict[Tuple[str, str], str]:
    return {(a, b): val for a, b, val in v}


CODECS: Dict[str, Tuple[Callable, Callable]] = {
    registry.PLAIN: (_plain, _plain),
    registry.OPS: (_enc_ops, _dec_ops),
    registry.OPS_INDEX: (_enc_ops_index, _dec_ops_index),
    registry.GOALS: (_enc_goals, _dec_goals),
    registry.PAIRS: (_enc_pairs, _dec_pairs),
    registry.PAIR_MAP: (_enc_pair_map, _dec_pair_map),
}


def _derive_subgoals(payload: Dict[str, Any]) -> List[str]:
    from arggym.core.scoring import subgoals_from

    return subgoals_from(payload["goals"], payload["base_ops"])


#: Scoring inputs that are recomputed from the row rather than stored.
DERIVATIONS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "subgoals": _derive_subgoals,
}


# -- writing -----------------------------------------------------------------

def encode_fields(fields: Sequence[Tuple[str, str]], item: Any,
                  base: Sequence[Operation]) -> Dict[str, Any]:
    """The declared fields of one item, as JSON-safe values."""
    out: Dict[str, Any] = {}
    for name, codec in fields:
        if not hasattr(item, name):
            raise MissingField(
                f"{type(item).__name__} has no attribute {name!r}, which the "
                f"registry says its rows carry; the item and the table disagree")
        out[name] = CODECS[codec][0](getattr(item, name), base)
    return out


# -- reading -----------------------------------------------------------------

def _require(d: Dict[str, Any], key: str, task: str, where: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise MissingField(
            f"{task}: this row has no metadata.{where}{key}, which scoring "
            f"{task} reads. A row that lost a field must refuse rather than "
            f"score with a default.")
    return d[key]


def decode_fields(fields: Sequence[Tuple[str, str]], stored: Dict[str, Any],
                  base: Sequence[Operation], task: str, where: str) -> Dict[str, Any]:
    return {name: CODECS[codec][1](_require(stored, name, task, where), base)
            for name, codec in fields}


def row_task(entry: Dict[str, Any]) -> str:
    """The registered task a row belongs to.

    `metadata.source_dataset` rather than `task`, because that is the key a
    composite dispatches on and the one that distinguishes the variants:
    `counter_argument_strict` and `counter_argument` are one module.
    """
    meta = entry.get("metadata")
    if not isinstance(meta, dict):
        raise MissingField("this row has no metadata, so nothing says what task it is")
    name = meta.get("source_dataset")
    if not name:
        raise MissingField(
            "this row has no metadata.source_dataset, so nothing says which task "
            "scores it. The top-level `task` is not a fallback: it names the row's "
            "task, while this names the registered one, and the two differ wherever "
            "a variant shares a module.")
    return name


def check_engine(want: Optional[str], task: str) -> None:
    """Refuse a row this engine would score differently.

    Scoring the six engine-checked tasks runs `python-argumentation` at scoring
    time, not only at generation time, so a frozen row is re-scorable against
    the pinned engine and no other. Returning a quietly different number is the
    failure this exists to prevent (`docs/dataset-contract.md` section 11).

    A row without the field predates it and is scored without the check.
    """
    if want is None:
        return
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        got = _pkg_version("python-argumentation")
    except PackageNotFoundError:  # pragma: no cover - depends on the install
        return
    if got != want:
        raise EngineMismatch(
            f"{task}: this row was built against python-argumentation {want} and "
            f"this environment has {got}. The score would be computed by a "
            f"different engine than the one that verified the gold.")


def score_input(entry: Dict[str, Any]) -> Tuple[registry.TaskSpec, Any]:
    """The row as the object its scorer takes.

    A dict for the six engine-checked tasks, whose scorer is `score_item`; an
    attribute bag for the other six, whose scorers take the item. The bag is not
    the item dataclass: a dataclass would need placeholder values for the fields
    scoring never reads, and a placeholder that a scorer someday reads is a
    wrong number, where a missing attribute is a traceback.
    """
    task = row_task(entry)
    spec = registry.get(task)
    meta = entry["metadata"]
    check_schema(_require(meta, "theory_schema", task, ""))
    check_engine(meta.get("pyarg_version"), task)

    theory: Dict[str, List[Operation]] = {
        f: ops_from_json(_require(meta, f, task, "")) for f in spec.theory_fields}
    # The theory a field indexes into, so a decoded `line_ops` holds the very
    # objects the scorer will see in `base_ops`.
    base: List[Operation] = theory[spec.theory_fields[0]] if spec.theory_fields else []

    fields: Dict[str, Any] = {"ordering": _require(meta, "ordering", task, "")}
    fields.update(decode_fields(spec.state_fields, meta.get("state", {}),
                                base, task, "state."))
    fields.update(decode_fields(spec.gold_fields, meta.get("gold", {}),
                                base, task, "gold."))

    if spec.scorer == registry.ENGINE:
        payload: Dict[str, Any] = {**theory, **fields}
        for key, value in spec.policy.items():
            payload[key] = (DERIVATIONS[key](payload)
                            if value == registry.DERIVED else value)
        return spec, payload
    return spec, SimpleNamespace(**theory, **fields)


def score(answer: str, entry: Dict[str, Any]) -> ScoreResult:
    """Score a model's text against one row, with no generator in sight."""
    spec, payload = score_input(entry)
    if spec.scorer == registry.ENGINE:
        from arggym.core.scoring import score_item

        return score_item(answer, payload)
    return import_module(f"arggym.tasks.{spec.module}").score(answer, payload)
