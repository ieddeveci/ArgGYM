import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import _answer_region
from evals.extract import extract, has_region


ANS = "[answer]\n1: justified\n[/answer]"


def score_region(result):
    """The text the scorer would actually read out of an extract() result."""
    return _answer_region(result["text"]).strip()


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


def test_revised_answer_wins_inside_a_single_field():
    """A model served without a reasoning parser puts draft and revision in one
    field. The revision is the submission; the abandoned draft is not."""
    cot = ("[answer]\n1: overruled\n[/answer]\n"
           "wait, no -- the attack is undercut\n"
           "[answer]\n1: justified\n[/answer]")
    r = extract(cot, "")
    assert r["has_region"]
    assert score_region(r) == "1: justified"


def test_unterminated_final_draft_falls_back_to_the_complete_region():
    """An answer region cut off mid-write is not a submission; the last complete
    one is."""
    r = extract("[answer]\n1: justified\n[/answer]\nhmm, let me redo it\n[answer]\n1: over", "")
    assert score_region(r) == "1: justified"


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
