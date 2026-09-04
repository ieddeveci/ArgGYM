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
from typing import Any, Dict, Optional, Tuple

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

# Which scorer a task's answer goes to. The six engine-checked tasks share
# `core.scoring.score_item`, which takes a dict; the other six have a `score` in
# their module, which takes the item.
ENGINE, MODULE = "engine", "module"

# How one field survives the trip through JSON. `plain` is a value JSON already
# holds; the rest name a shape JSON drops. `ops_index` is a sublist of the
# theory written as positions in it, `pairs` and `pair_map` are the tuples
# `semantics_query` keys its gold by, and `goals` keeps the two keys the scorer
# reads. The functions live in `core/rows.py`.
PLAIN = "plain"
OPS = "ops"
OPS_INDEX = "ops_index"
GOALS = "goals"
PAIRS = "pairs"
PAIR_MAP = "pair_map"

#: A policy value the scoring path recomputes from the row instead of reading.
#: Nothing derived is serialized, or the copy drifts from the definition.
DERIVED = "<derived>"


@dataclass(frozen=True)
class TaskSpec:
    """One registered task, including its variant.

    The field tables are what makes a frozen row scorable: between them they
    name everything the task's scorer reads, so `core/rows.py` can write a row
    and read it back without a branch per task.
    """

    name: str
    module: str
    checker: str
    answer_shape: str
    #: Passed to the module's `make_item` on top of (level, seed, ordering).
    variant: Dict[str, Any] = field(default_factory=dict)
    #: Item attributes holding the theory the question shows. `perturbation`
    #: renders two lists, so this is a tuple rather than one name.
    theory_fields: Tuple[str, ...] = ("base_ops",)
    #: Scoring inputs the question already gives away, as (attribute, codec).
    #: They go to `metadata.state`.
    state_fields: Tuple[Tuple[str, str], ...] = ()
    #: Scoring inputs that give the answer away, as (attribute, codec). They go
    #: to `metadata.gold`, so "do not show the model `metadata.gold`" is one
    #: rule rather than twelve. `formalization`'s reference_ops is the gold
    #: answer rather than a theory, and `min_directives` is here because the
    #: prompt never says how many directives an answer needs.
    gold_fields: Tuple[Tuple[str, str], ...] = ()
    #: Which scorer takes the answer.
    scorer: str = MODULE
    #: Scoring policy that is constant for the task rather than per item, so it
    #: comes from this table and is not stored in the row. A `DERIVED` value is
    #: recomputed from the row.
    policy: Dict[str, Any] = field(default_factory=dict)

    def make_item(self, level: int, seed: int, ordering: str,
                  **kwargs: Any) -> Optional[Any]:
        mod = import_module(f"arggym.tasks.{self.module}")
        return mod.make_item(level, seed, ordering, **{**self.variant, **kwargs})


#: On every task. `item.metadata` is the generator's own statistics, and most of
#: it describes the reference: defeat_diagnosis records the gold status the
#: answer must state, status_query the exact label distribution, and five tasks
#: the answer's length. An allowlist of the safe keys would leak the first one
#: somebody forgot, so the whole blob is gold by default.
_STATS = ("metadata", PLAIN)


def _spec(name, module, checker, shape, theory=("base_ops",), state=(), gold=(),
          scorer=MODULE, policy=None, **variant) -> Tuple[str, TaskSpec]:
    return name, TaskSpec(name, module, checker, shape, variant, theory, state, tuple(gold) + (_STATS,),
                          scorer, dict(policy or {}))


#: What the six engine-checked tasks share: the goals are in the question, the
#: minimal directive count is not.
_CONSTRUCTION = dict(state=(("goals", GOALS),), gold=(("min_directives", PLAIN),),
                     scorer=ENGINE)

REGISTRY: Dict[str, TaskSpec] = dict([
    _spec("preference_construction", "preference_construction", VERIFIED, OPERATION_LIST,
          **_CONSTRUCTION, policy={"allow_strict": False, "preferences_only": True}),
    _spec("counter_argument", "counter_argument", VERIFIED, OPERATION_LIST,
          **_CONSTRUCTION, policy={"allow_strict": False}, allow_strict=False),
    _spec("counter_argument_strict", "counter_argument", VERIFIED, OPERATION_LIST,
          **_CONSTRUCTION, policy={"allow_strict": True}, allow_strict=True),
    # subgoals is a property over the goals and the theory, so the row records
    # that these three use it and the scoring path recomputes it.
    _spec("attack", "attack_defense", VERIFIED, OPERATION_LIST,
          **_CONSTRUCTION, policy={"subgoals": DERIVED}, mode="attack"),
    _spec("defence", "attack_defense", VERIFIED, OPERATION_LIST,
          **_CONSTRUCTION, policy={"subgoals": DERIVED}, mode="defence"),
    _spec("attack_defense", "attack_defense", VERIFIED, OPERATION_LIST,
          **_CONSTRUCTION, policy={"subgoals": DERIVED}, mode="attack_defense"),
    # The question is prose and the operations are the gold answer, so this
    # task has no theory to publish and everything op-shaped is gold.
    _spec("formalization", "formalization", GRADED, OPERATION_LIST,
          theory=(), state=(("queried", PLAIN),),
          gold=(("reference_ops", OPS), ("gold_status", PLAIN))),
    _spec("status_query", "status_query", GRADED, LABEL_MAP,
          state=(("queried", PLAIN),), gold=(("gold", PLAIN),)),
    _spec("semantics_query", "semantics_query", GRADED, LABEL_MAP,
          state=(("queries", PAIRS),), gold=(("gold", PAIR_MAP),)),
    # survivors names the claims that did not change, so it is gold too.
    _spec("perturbation", "perturbation", GRADED, LABEL_MAP,
          theory=("base_ops", "pert_ops"),
          gold=(("gold", PLAIN), ("survivors", PLAIN))),
    # line_ops is the justifying line the answer must reproduce, and it is a
    # sublist of base_ops, so it is written as positions in the theory.
    _spec("claim_chain", "claim_chain", GRADED, ORDERED_SEQUENCE,
          state=(("claim", PLAIN),), gold=(("line_ops", OPS_INDEX),)),
    _spec("defeat_diagnosis", "defeat_diagnosis", GRADED, RECORD_LIST,
          gold=(("claim_status", PLAIN), ("diagnoses", PLAIN))),
])


def task_names() -> Tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def get(name: str) -> TaskSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known: {', '.join(task_names())}") from None
