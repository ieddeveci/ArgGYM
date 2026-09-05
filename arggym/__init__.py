"""ArgGYM - a generator and grader for defeasible-argumentation tasks in ASPIC+.

The whole adoption surface is here. Everything below is importable without
pulling a task module, so `import arggym` stays cheap.

ArgGYM owns **what a legal answer is**. You own **how you get one**. Write a
solver -- a bare model, an agent with tools, a symbolic procedure, anything --
that turns a question into an answer, and hand the answer over:

    import arggym

    ds = arggym.create("status_query", level=9,
                       ordering="weakest_link_elitist", size=100)
    entry = ds[0]
    answer = my_solver(entry["question"])     # your prompt, your parsing
    result = ds.score(answer, entry)

The question states the task and what a legal answer must contain. It says
nothing about where to put the answer, because composing the prompt and pulling
the answer back out of a completion are the harness's job, not the dataset's.
That is what lets any convention work: XML tags, a boxed expression, a JSON
schema, constrained decoding, or a solver that returns the answer directly.

If you would rather the question carried the delivery sentence, ask for one and
it renders exactly one sentence more:

    ds = arggym.create("status_query", level=9, template=arggym.XML_TAGS)
    text = my_model(ds[0]["question"])
    result = ds.score(arggym.extract_answer(text), ds[0])

An answer need not be text either. Every task takes the value directly, so a
solver with a JSON schema or constrained decoding submits it and gets the same
`ScoreResult` back:

    result = ds.score_value({"ab1": "justified"}, entry)

A row carries everything its scorer reads, so a line read back from a frozen
JSONL scores without a dataset object at all:

    result = arggym.score_row(answer, json.loads(line))
    result = arggym.score_row_value(value, json.loads(line))

See `docs/dataset-contract.md`.
"""

__version__ = "2.0.0"

from arggym.core.answers import (
    DEFAULT_TEMPLATE,
    XML_TAGS,
    AnswerTemplate,
    ScoreResult,
    UnparseableAnswer,
    extract_answer,
)
from arggym.core.dataset import BuildFailed, ConcatDataset, TaskDataset, create, from_spec
from arggym.core.floors import corrected, floors
from arggym.core.registry import TaskSpec, task_names
from arggym.core.registry import get as get_task
from arggym.core.rows import MissingField
from arggym.core.rows import score as score_row
from arggym.core.rows import score_value as score_row_value
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
    "extract_answer", "ScoreResult", "UnparseableAnswer",
    "AnswerTemplate", "XML_TAGS", "DEFAULT_TEMPLATE",
    # reading a score
    "floors", "corrected", "score_row", "score_row_value", "MissingField",
    # tasksets
    "TasksetSpec", "SeedPolicy", "load_spec",
    # serialization
    "ops_to_json", "ops_from_json", "THEORY_SCHEMA",
]
