import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.extract import extract, has_region


ANS = "[answer]\n1: justified\n[/answer]"


def test_answer_in_content_is_used_directly():
    r = extract(f"reasoning here\n{ANS}", "some chain of thought")
    assert r["has_region"] and not r["used_fallback"]
    assert "justified" in r["text"]


def test_content_wins_over_cot_when_both_have_answers():
    """A model that drafts an answer mid-CoT then revises must be scored on the
    revision, not the draft."""
    cot = "[answer]\n1: overruled\n[/answer] wait, that's wrong"
    r = extract(ANS, cot)
    assert not r["used_fallback"]
    assert r["answer_in_cot"] is True      # recorded...
    assert "justified" in r["text"]        # ...but the content answer is scored
    assert "overruled" not in r["text"]


def test_falls_back_to_cot_when_content_empty():
    r = extract("", f"thinking\n{ANS}")
    assert r["used_fallback"] and r["has_region"] and r["answer_in_cot"]
    assert "justified" in r["text"]


def test_no_region_anywhere():
    r = extract("I think it is justified.", "")
    assert not r["has_region"] and not r["answer_in_cot"]


def test_none_inputs_do_not_crash():
    r = extract(None, None)
    assert r["text"] == "" and not r["has_region"]


def test_has_region_requires_both_tags():
    assert not has_region("[answer] 1: justified")
    assert has_region("[answer] 1: justified [/answer]")
