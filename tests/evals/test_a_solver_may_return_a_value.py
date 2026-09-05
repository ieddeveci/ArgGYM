"""A solver that never wrote text still gets scored.

Constrained decoding, a JSON schema, a tool call: a solver may produce the answer
as a value, and asking it to render ArgGYM's DSL so the harness can parse it back
would be the dataset dictating a serialization
(`docs/dataset-contract.md` section 4). Setting `Attempt.value` routes scoring to
`arggym.score_row_value` and skips extraction entirely.
"""
from __future__ import annotations

import os

import arggym
from evals import artifacts, values
from evals.run import generate
from evals.score import score_one, score_run
from evals.types import Attempt


def value_for(row):
    """The row's own reference answer, as a JSON-carried value of its shape."""
    from arggym.core.scoring import parse

    ref = row["reference_answer"]
    if row["metadata"]["answer_shape"] == "operation_list":
        return values.encode(parse(ref, {}), row)
    return values.encode({k.strip(): v.strip() for k, _, v in
                          (line.partition(":") for line in ref.splitlines() if line)},
                         row)


def test_a_value_answer_scores_the_same_as_its_text(rows):
    for row in rows:
        by_text = arggym.score_row(row["reference_answer"], row)
        record = score_one({"id": row["id"], "task": row["task"], "level": 3,
                            "ordering": "last_link_elitist", "completion": "",
                            "value": value_for(row)}, row, "xml_tags")
        assert record["score"] == by_text.score == 1.0, row["task"]
        # Nothing was extracted, so the flags that describe extraction say so
        # rather than reporting a missing fence.
        assert record["no_answer_region"] is False
        assert record["answer_in_cot"] is False


def test_a_value_solver_needs_no_endpoint(tmp_path, rows, taskset_file):
    """The seam takes any callable. This one has no model in it at all."""
    def perfect(row):
        return Attempt(value=value_for(row), completion="")

    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    manifest, _ = __import__("evals.taskset", fromlist=["load"]).load(taskset_file)
    artifacts.write_json(os.path.join(run_dir, artifacts.RUN), {
        "status": "completed", "taskset": taskset_file,
        "taskset_hash": manifest["taskset_hash"], "template": "xml_tags"})
    generate(rows, perfect, run_dir, concurrency=2, progress=False)

    metrics = score_run(run_dir)
    assert metrics["coverage"]["n_scored"] == len(rows)
    for task, v in metrics["by_task"].items():
        assert v["mean"] == 1.0, task


def parsed_value(row):
    """The reference answer as the live value its scorer expects.

    This reaches into the package on purpose. The four answer shapes include two
    a harness cannot construct from the public API alone -- `semantics_query`
    keys its map by a tuple, and an operation list is a list of `Operation`
    objects -- and the encoding in `evals/values.py` exists precisely for them.
    Testing it on anything less than all twelve tasks would leave the shapes it
    was written for uncovered.
    """
    from importlib import import_module

    from arggym.core import registry
    from arggym.core.rows import score_input

    spec, payload = score_input(row)
    if spec.scorer == registry.ENGINE:
        from arggym.core.scoring import parse
        return parse(row["reference_answer"], payload)
    return import_module(f"arggym.tasks.{spec.module}").parse(
        row["reference_answer"], payload)


def test_every_answer_shape_survives_the_trip_through_json():
    """Generating and scoring are separate programs, so a value is JSON in between."""
    import json

    for task in sorted(arggym.task_names()):
        row = arggym.TaskDataset(task, 3, "last_link_elitist", size=1, seed=0)[0]
        value = parsed_value(row)
        # Exactly what `run.py` writes and `score.py` reads back.
        carried = json.loads(json.dumps(values.encode(value, row), default=str))
        result = arggym.score_row_value(values.decode(carried, row), row)
        assert result.score == 1.0, (task, row["metadata"]["answer_shape"], result)
