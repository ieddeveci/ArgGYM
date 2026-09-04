"""ArgGYM - a generator and grader for defeasible-argumentation tasks in ASPIC+.

The whole adoption surface is here. Everything below is importable without
pulling a task module, so `import arggym` stays cheap.

    import arggym

    ds = arggym.create("status_query", level=9,
                       ordering="weakest_link_elitist", size=100)
    entry = ds[0]                       # question / reference_answer / metadata
    text = my_model(entry["question"])  # your prompt template, your delimiters
    result = ds.score(arggym.extract_answer(text), entry)

`extract_answer` is a convenience, not a contract: ArgGYM accepts a bare answer
body, so an evaluator that already knows where its answer ends can skip it.
See `docs/dataset-contract.md`.
"""

__version__ = "2.0.0"

from arggym.core.answers import ScoreResult, extract_answer
from arggym.core.dataset import (BuildFailed, ConcatDataset, TaskDataset,
                                 create, from_spec)
from arggym.core.registry import TaskSpec, get as get_task, task_names
from arggym.core.serialize import THEORY_SCHEMA, ops_from_json, ops_to_json
from arggym.core.spec import SeedPolicy, TasksetSpec, load as load_spec

__all__ = [
    "__version__",
    # tasks
    "task_names", "get_task", "TaskSpec",
    # items
    "create", "from_spec", "TaskDataset", "ConcatDataset", "BuildFailed",
    # answers
    "extract_answer", "ScoreResult",
    # tasksets
    "TasksetSpec", "SeedPolicy", "load_spec",
    # serialization
    "ops_to_json", "ops_from_json", "THEORY_SCHEMA",
]
