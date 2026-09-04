"""One adapter per mode, so a property test can run the same code over all twelve.

Each adapter mirrors how `arggym/inspector.py` builds and scores that mode, and how
`arggym/core/export.py` exports it. Read the task module before changing a field here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from arggym.aspic.engine import Operation
from arggym.core.answers import ScoreResult
from arggym.core.scoring import score_item
from arggym.tasks import (attack_defense, claim_chain, counter_argument, defeat_diagnosis,
                          formalization, perturbation, preference_construction,
                          semantics_query, status_query)

LABEL_MAP, CONSTRUCTION, DIAGNOSIS, CHAIN, FORMALIZATION = (
    "label_map", "construction", "diagnosis", "chain", "formalization")
FAMILIES = (LABEL_MAP, CONSTRUCTION, DIAGNOSIS, CHAIN, FORMALIZATION)

# The reason each scorer gives an answer that carries one stray token. The shared
# construction scorer, formalization and perturbation count lines; the others count tokens.
UNPARSEABLE_LINES = "unparseable_lines"
UNPARSEABLE_TOKENS = "unparseable_tokens"


@dataclass(frozen=True)
class Adapter:
    family: str
    make: Callable[[int, int, str], Any]          # (level, seed, ordering) -> item | None
    reference: Callable[[Any], str]
    score: Callable[[str, Any], Dict]
    theory_ops: Callable[[Any], List[Operation]]  # the theory the prompt shows
    prompt: Callable[[Any], str]
    junk_reason: str                               # UNPARSEABLE_LINES or UNPARSEABLE_TOKENS

    def __post_init__(self):
        """`adapter.score` always yields a mapping.

        The scorers are mid-migration to `ScoreResult`, so a task returns either shape.
        Normalizing here keeps every test written against one shape and means converting
        a task module needs no edit in this file.
        """
        object.__setattr__(self, "score", _as_mapping(self.score))


def _as_mapping(fn: Callable[[str, Any], Any]) -> Callable[[str, Any], Dict]:
    def scored(text: str, item: Any) -> Dict:
        result = fn(text, item)
        return result.as_dict() if isinstance(result, ScoreResult) else result
    return scored


def _attack_defense(mode: str) -> Adapter:
    return Adapter(
        family=CONSTRUCTION,
        make=lambda level, seed, ordering: attack_defense.make_item(level, seed, ordering, mode),
        reference=lambda it: it.reference,
        score=lambda text, it: score_item(text, it.as_score_input()),
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_LINES)


def _counter_argument(allow_strict: bool) -> Adapter:
    return Adapter(
        family=CONSTRUCTION,
        make=lambda level, seed, ordering: counter_argument.make_item(
            level, seed, ordering, allow_strict=allow_strict),
        reference=lambda it: it.reference,
        score=lambda text, it: score_item(text, counter_argument.as_score_input(it)),
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_LINES)


MODES: Dict[str, Adapter] = {
    "semantics_query": Adapter(
        family=LABEL_MAP,
        make=semantics_query.make_item,
        reference=lambda it: it.reference,
        score=semantics_query.score,
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_TOKENS),
    "status_query": Adapter(
        family=LABEL_MAP,
        make=status_query.make_item,
        reference=lambda it: it.reference,
        score=status_query.score,
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_TOKENS),
    # The formalization prompt is natural language; the theory it encodes is reference_ops.
    "formalization": Adapter(
        family=FORMALIZATION,
        make=formalization.make_item,
        reference=lambda it: it.reference,
        score=formalization.score,
        theory_ops=lambda it: it.reference_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_LINES),
    "defeat_diagnosis": Adapter(
        family=DIAGNOSIS,
        make=defeat_diagnosis.make_item,
        reference=lambda it: it.reference,
        score=defeat_diagnosis.score,
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_TOKENS),
    "claim_chain": Adapter(
        family=CHAIN,
        make=claim_chain.make_item,
        reference=lambda it: it.reference,
        score=claim_chain.score,
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_TOKENS),
    "preference_construction": Adapter(
        family=CONSTRUCTION,
        make=preference_construction.make_item,
        reference=lambda it: it.reference,
        score=lambda text, it: score_item(text, it.as_score_input()),
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_LINES),
    "counter_argument_strict": _counter_argument(allow_strict=True),
    "counter_argument": _counter_argument(allow_strict=False),
    # PerturbItem.reference is a method, not a field.
    "perturbation": Adapter(
        family=LABEL_MAP,
        make=perturbation.make_item,
        reference=lambda it: it.reference(),
        score=perturbation.score,
        theory_ops=lambda it: it.base_ops,
        prompt=lambda it: it.prompt,
        junk_reason=UNPARSEABLE_LINES),
    "attack": _attack_defense("attack"),
    "defence": _attack_defense("defence"),
    "attack_defense": _attack_defense("attack_defense"),
}
