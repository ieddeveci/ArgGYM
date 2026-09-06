"""The example an adopter copies has to run.

`docs/dataset-card.md` and #54 promise a path from install to a scored JSONL.
The model call is stubbed; everything else -- reading the frozen file, composing
the prompt, pulling the answer out of the completion, scoring a row, and the
report -- is exercised.

This file is also where the boundary is checked from the outside: the three jobs
the example does are the three the dataset does not.
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


def test_the_example_names_one_convention_and_uses_it_for_both_halves(taskset):
    """Stating a fence and reading one back are two chances to disagree.

    The example used to carry its own tag regex alongside the package's. Here the
    convention is named once and both halves are derived from it, so a reader who
    swaps `TEMPLATE` gets an instruction and an extractor that still match.
    """
    src = (ROOT / "examples" / "evaluate.py").read_text()
    assert "arggym.extract_answer(" in src
    assert "re.compile" not in src, "the example defines a second answer-region regex"
    assert evaluate.prompt_for("Q") == f"Q\n{evaluate.TEMPLATE.instruction}"
    assert evaluate.answer_from(evaluate.TEMPLATE.wrap("body")) == "body"


def test_no_question_in_the_taskset_says_where_to_put_the_answer(taskset):
    """The dataset states the task. Where the answer goes is the harness's sentence."""
    from arggym.core.answers import DEFAULT_TEMPLATE
    rows = list(evaluate.rows(str(taskset)))
    assert rows
    for row in rows:
        assert DEFAULT_TEMPLATE.instruction not in row["question"]
        assert DEFAULT_TEMPLATE.open not in row["question"]
        assert "answer_template" not in row["metadata"]


def test_the_prompt_the_example_sends_does_say_where(taskset):
    rows = list(evaluate.rows(str(taskset)))
    prompt = evaluate.prompt_for(rows[0]["question"])
    assert prompt.startswith(rows[0]["question"])
    assert prompt.endswith(evaluate.TEMPLATE.instruction)


def test_a_perfect_model_scores_one_on_every_row(taskset, monkeypatch):
    def stub(base_url, model, key, prompt, max_tokens, timeout):
        # Reason out loud, then answer with the reference for whichever row carries
        # this prompt. The reasoning is the reason the example extracts at all.
        for row in evaluate.rows(str(taskset)):
            if prompt == evaluate.prompt_for(row["question"]):
                return {"text": "Let me work through it.\n"
                                + evaluate.TEMPLATE.wrap(row["reference_answer"]),
                        "finish_reason": "stop"}
        raise AssertionError("prompt does not match any row")

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


def test_one_unscorable_row_does_not_cost_the_whole_run(taskset, monkeypatch):
    # Every completion has already been paid for by the time scoring runs, so
    # losing them all to an unopened output file is the expensive failure.
    def stub(base_url, model, key, question, max_tokens, timeout):
        return {"text": "<answer>\nnonsense\n</answer>", "finish_reason": "stop"}

    def boom(text, row):
        raise RuntimeError("this row was built against another engine")

    monkeypatch.setattr(evaluate, "complete", stub)
    monkeypatch.setattr(evaluate.arggym, "score_row", boom)
    out = taskset.parent / "unscorable.jsonl"
    monkeypatch.setattr(sys, "argv",
                        ["evaluate.py", str(taskset), "--model", "stub", "--out", str(out)])
    assert evaluate.main() == 0
    scored = [json.loads(x) for x in open(out)]
    assert len(scored) == 4
    assert all("unscorable row" in s["error"] for s in scored)
    assert all(s["completion"] for s in scored), "the completions must be kept"


def test_a_malformed_api_response_is_an_error_not_a_crash(monkeypatch):
    class _Resp:
        def read(self): return b'{"not_choices": []}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(evaluate.json, "load", lambda r: {"not_choices": []})
    monkeypatch.setattr(evaluate.urllib.request, "urlopen", lambda *a, **k: _Resp())
    got = evaluate.complete("http://x/v1", "m", "k", "q", 10, 1)
    assert "malformed response" in got["error"]
