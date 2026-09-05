"""An adopter should never need a private name.

Until now `arggym/__init__.py` exported only `__version__`, so the task list was
the private `export._EXPORTABLE` and every scoring path went through a task
module. That is what makes people fork instead of import.
"""
import arggym


def test_everything_advertised_is_actually_there():
    missing = [n for n in arggym.__all__ if not hasattr(arggym, n)]
    assert not missing, missing


def test_importing_the_package_does_not_pull_a_task_module():
    # Twelve task modules import the engine and the curriculum. Paying for all
    # of them at `import arggym` would make the package unpleasant to depend on.
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, arggym; "
         "print([m for m in sys.modules if m.startswith('arggym.tasks.')])"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_a_whole_evaluation_runs_from_the_public_names_alone():
    ds = arggym.create("claim_chain", level=3, ordering="last_link_elitist", size=2)
    assert len(ds) == 2
    entry = ds[0]
    assert set(entry) == {"id", "task", "question", "reference_answer", "metadata"}
    assert entry["question"] and entry["reference_answer"]
    # A model's completion arrives however the evaluator chose to wrap it; the
    # body is what gets scored either way.
    assert arggym.extract_answer("thinking...\n<answer>\nBODY\n</answer>").strip() == "BODY"
    assert arggym.extract_answer("BODY") == "BODY"


def test_a_row_can_be_rebuilt_into_operations_without_the_generator():
    entry = arggym.create("status_query", level=3, size=1)[0]
    ops = arggym.ops_from_json(entry["metadata"]["base_ops"])
    assert ops and all(o.kind for o in ops)


def test_the_task_list_is_public():
    assert "status_query" in arggym.task_names()
    assert arggym.get_task("status_query").checker == "graded"


def test_a_solver_may_hand_over_a_value_instead_of_text():
    """The README shows this, so it has to work from the public surface alone.

    A solver with a JSON schema or constrained decoding produces the answer; asking
    it to render our DSL so we can parse it back would be the dataset dictating a
    serialization (`docs/dataset-contract.md` section 4).
    """
    ds = arggym.create("status_query", level=3, ordering="last_link_elitist", size=1)
    entry = ds[0]
    from_value = ds.score_value(dict(entry["metadata"]["gold"]["gold"]), entry)
    from_text = ds.score(entry["reference_answer"], entry)
    assert from_value.as_dict() == from_text.as_dict()
    assert from_value.score == 1.0 and from_value.success is True


def test_a_frozen_row_scores_from_a_value_with_no_dataset_object():
    ds = arggym.create("status_query", level=3, ordering="last_link_elitist", size=1)
    entry = ds[0]
    assert arggym.score_row_value(dict(entry["metadata"]["gold"]["gold"]),
                                  entry).score == 1.0
