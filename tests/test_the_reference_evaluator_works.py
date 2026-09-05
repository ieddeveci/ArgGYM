"""The example an adopter copies has to run.

`docs/dataset-card.md` and #54 promise a path from install to a scored JSONL.
The model call is stubbed; everything else -- reading the frozen file, finding
the answer inside the fence the question asked for, scoring a row, and the
report -- is exercised.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from arggym.core.freeze import freeze
from arggym.core.spec import SeedPolicy, TasksetSpec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import evaluate  # noqa: E402


@pytest.fixture(scope="module")
def taskset(tmp_path_factory):
    spec = TasksetSpec(tasks=("claim_chain", "status_query"), levels=(3,),
                       orderings=("last_link_elitist",),
                       seeds=SeedPolicy(take=2, scan_limit=10))
    path = tmp_path_factory.mktemp("ts") / "taskset.jsonl"
    freeze(spec, str(path), verbose=False)
    return path


def test_the_manifest_line_is_not_read_as_an_item(taskset):
    got = list(evaluate.rows(str(taskset)))
    assert len(got) == 4
    assert all("question" in r for r in got)


def test_the_example_reads_the_fence_the_question_asked_for(taskset):
    """It does not define a second one.

    The example used to carry its own tag regex, arrived at independently and
    identical to the package's. Two copies of the rule that says where an answer
    ends is how they come to disagree, so the example reads the region with
    `arggym.extract_answer` and its own text says which convention that is.
    """
    src = (ROOT / "examples" / "evaluate.py").read_text()
    assert "arggym.extract_answer(" in src
    assert "re.compile" not in src, "the example defines a second answer-region regex"


def test_every_question_in_the_taskset_names_the_fence(taskset):
    from arggym.core.answers import DEFAULT_TEMPLATE
    rows = list(evaluate.rows(str(taskset)))
    for row in rows:
        assert DEFAULT_TEMPLATE.instruction in row["question"]
        assert row["metadata"]["answer_template"] == DEFAULT_TEMPLATE.name


def test_a_perfect_model_scores_one_on_every_row(taskset, monkeypatch):
    def stub(base_url, model, key, question, max_tokens, timeout):
        # Answer with the reference for whichever row carries this question.
        for row in evaluate.rows(str(taskset)):
            if row["question"] == question:
                return {"text": "<answer>\n" + row["reference_answer"] + "\n</answer>",
                        "finish_reason": "stop"}
        raise AssertionError("question not in the taskset")

    monkeypatch.setattr(evaluate, "complete", stub)
    monkeypatch.setattr(sys, "argv",
                        ["evaluate.py", str(taskset), "--model", "stub",
                         "--out", str(taskset.parent / "scored.jsonl")])
    assert evaluate.main() == 0
    scored = [json.loads(x) for x in open(taskset.parent / "scored.jsonl")]
    assert len(scored) == 4
    assert all(s["score"] == 1.0 and s["success"] for s in scored), scored


def test_an_api_failure_is_not_recorded_as_a_wrong_answer(taskset, monkeypatch):
    # Collapsing the two would report infrastructure trouble as a reasoning
    # result, which is the whole reason the field is separate.
    monkeypatch.setattr(evaluate, "complete",
                        lambda *a, **k: {"error": "URLError: refused"})
    monkeypatch.setattr(sys, "argv",
                        ["evaluate.py", str(taskset), "--model", "stub",
                         "--out", str(taskset.parent / "failed.jsonl")])
    assert evaluate.main() == 0
    scored = [json.loads(x) for x in open(taskset.parent / "failed.jsonl")]
    assert all(s["error"] and "score" not in s for s in scored)


def test_the_report_refuses_to_print_one_number(capsys):
    evaluate.report([
        {"task": "status_query", "score": 0.9, "success": True, "error": None,
         "truncated": False},
        {"task": "claim_chain", "score": 0.1, "success": False, "error": None,
         "truncated": False},
    ])
    out = capsys.readouterr().out
    assert "status_query" in out and "claim_chain" in out
    assert "No overall mean" in out
    assert "0.500" not in out, "an unweighted mean over incomparable metrics"


def test_the_example_is_syntactically_a_script():
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "evaluate.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--base-url" in r.stdout
