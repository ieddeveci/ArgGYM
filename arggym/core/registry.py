"""The task list, as data.

`core/export.py` reaches into a private dict and then dispatches through an
if/elif ladder that restates each task's shape. That design's default failure is
forgetting a branch, which is how two tasks went missing from the export (#23),
and it is why nobody can write "for each task: generate, prompt, score" without
special-casing every module.

Three tasks are variants rather than modules: `counter_argument_strict` differs
from `counter_argument` by a flag, and `attack`, `defence` and `attack_defense`
are three modes of one builder. The variant is part of the task identity, so
they are registered separately and the adapter holds the difference.

Modules are imported when a task is first built. Importing the registry costs
nothing, so `import arggym` stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple

# How an answer is judged. `exact` means a normalized comparison against the
# reference is sound; `graded` a continuous scorer over a unique gold; and
# `verified` that the engine is run on theory plus answer. No task is `exact`
# today -- see docs/dataset-contract.md section 2.
EXACT, GRADED, VERIFIED = "exact", "graded", "verified"

# The four shapes a parsed answer takes. `ordered_sequence` is separate from a
# selection set because claim_chain scores the order, and `record_list` because
# defeat_diagnosis answers with structured records rather than labels.
OPERATION_LIST = "operation_list"
LABEL_MAP = "label_map"
ORDERED_SEQUENCE = "ordered_sequence"
RECORD_LIST = "record_list"


@dataclass(frozen=True)
class TaskSpec:
    """One registered task, including its variant."""

    name: str
    module: str
    checker: str
    answer_shape: str
    #: Passed to the module's `make_item` on top of (level, seed, ordering).
    variant: Dict[str, Any] = field(default_factory=dict)
    #: Item attributes holding the theory the question shows. `perturbation`
    #: renders two lists, so this is a tuple rather than one name.
    theory_fields: Tuple[str, ...] = ("base_ops",)
    #: Item attributes holding operations that give the answer away.
    #: `formalization`'s reference_ops is the gold answer, not a theory.
    gold_op_fields: Tuple[str, ...] = ()

    def make_item(self, level: int, seed: int, ordering: str,
                  **kwargs: Any) -> Optional[Any]:
        mod = import_module(f"arggym.tasks.{self.module}")
        return mod.make_item(level, seed, ordering, **{**self.variant, **kwargs})


def _spec(name, module, checker, shape, theory=("base_ops",), gold=(),
          **variant) -> Tuple[str, TaskSpec]:
    return name, TaskSpec(name, module, checker, shape, variant, theory, gold)


REGISTRY: Dict[str, TaskSpec] = dict([
    _spec("preference_construction", "preference_construction", VERIFIED, OPERATION_LIST),
    _spec("counter_argument", "counter_argument", VERIFIED, OPERATION_LIST,
          allow_strict=False),
    _spec("counter_argument_strict", "counter_argument", VERIFIED, OPERATION_LIST,
          allow_strict=True),
    _spec("attack", "attack_defense", VERIFIED, OPERATION_LIST, mode="attack"),
    _spec("defence", "attack_defense", VERIFIED, OPERATION_LIST, mode="defence"),
    _spec("attack_defense", "attack_defense", VERIFIED, OPERATION_LIST,
          mode="attack_defense"),
    # The question is prose and the operations are the gold answer, so this
    # task has no theory to publish and everything op-shaped is gold.
    _spec("formalization", "formalization", GRADED, OPERATION_LIST,
          theory=(), gold=("reference_ops",)),
    _spec("status_query", "status_query", GRADED, LABEL_MAP),
    _spec("semantics_query", "semantics_query", GRADED, LABEL_MAP),
    _spec("perturbation", "perturbation", GRADED, LABEL_MAP,
          theory=("base_ops", "pert_ops")),
    # line_ops is the justifying line the answer must reproduce.
    _spec("claim_chain", "claim_chain", GRADED, ORDERED_SEQUENCE,
          gold=("line_ops",)),
    _spec("defeat_diagnosis", "defeat_diagnosis", GRADED, RECORD_LIST),
])


def task_names() -> Tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def get(name: str) -> TaskSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known: {', '.join(task_names())}") from None
