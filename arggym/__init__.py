"""ArgGYM - a generator and grader for defeasible-argumentation tasks in ASPIC+.

The whole adoption surface is here. Everything below is importable without
pulling a task module, so `import arggym` stays cheap.

    import arggym

    ds = arggym.create("status_query", level=9,
                       ordering="weakest_link_elitist", size=100)
    entry = ds[0]                       # question / reference_answer / metadata
    text = my_model(entry["question"])  # the question asks for <answer> tags
    result = ds.score(arggym.extract_answer(text), entry)

A row carries everything its scorer reads, so a line read back from a frozen
JSONL scores without a dataset object at all:

    result = arggym.score_row(text, json.loads(line))

The question asks for the answer between `<answer>` and `</answer>`, which is
what `extract_answer` reads; it also still reads the older `[answer]` pair, and
returns a completion whole when it is already just the answer. Which fence a
question asks for is a render-time choice: `create(..., template=SQUARE_TAGS)`
re-renders rather than editing prompt strings, and the row records which one it
used. See `docs/dataset-contract.md`.
"""

__version__ = "2.0.0"

from arggym.core.answers import (
    DEFAULT_TEMPLATE,
    SQUARE_TAGS,
    XML_TAGS,
    AnswerTemplate,
    ScoreResult,
    extract_answer,
)
from arggym.core.dataset import BuildFailed, ConcatDataset, TaskDataset, create, from_spec
from arggym.core.floors import corrected, floors
from arggym.core.registry import TaskSpec, task_names
from arggym.core.registry import get as get_task
from arggym.core.rows import MissingField
from arggym.core.rows import score as score_row
from arggym.core.serialize import THEORY_SCHEMA, ops_from_json, ops_to_json
from arggym.core.spec import SeedPolicy, TasksetSpec
from arggym.core.spec import load as load_spec

__all__ = [
    "__version__",
    # tasks
    "task_names", "get_task", "TaskSpec",
    # items
    "create", "from_spec", "TaskDataset", "ConcatDataset", "BuildFailed",
    # answers
    "extract_answer", "ScoreResult", "score_row",
    "AnswerTemplate", "XML_TAGS", "SQUARE_TAGS", "DEFAULT_TEMPLATE",
    # reading a score
    "floors", "corrected", "score_row", "MissingField",
    # tasksets
    "TasksetSpec", "SeedPolicy", "load_spec",
    # serialization
    "ops_to_json", "ops_from_json", "THEORY_SCHEMA",
]
