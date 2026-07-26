import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import template
from evals.scoring import _agg, aggregate, score_sample
from evals.taskset import build_rows


def _row():
    return build_rows(["status_query"], ["symbolic"], [1], 1, 20260721)[0]


def _gen(**kw):
    g = {"content": "", "reasoning_content": "", "finish_reason": "stop",
         "usage": {"completion_tokens": 100}, "latency_s": 1.0, "error": None}
    g.update(kw)
    return g


def test_perfect_answer_scores_one():
    row = _row()
    # Gold is stored raw; a compliant model submits it inside the answer template.
    s = score_sample(row, _gen(content=template.wrap(row["entry"]["answer"])))
    assert s["score"] == 1.0
    assert not s["no_answer_region"] and not s["zero_score_with_valid_region"]


def test_missing_answer_region_is_flagged_not_silently_zero():
    s = score_sample(_row(), _gen(content="I reasoned but forgot the tags."))
    assert s["score"] == 0.0
    assert s["no_answer_region"] is True
    assert s["zero_score_with_valid_region"] is False


def test_wrong_but_wellformed_answer_flags_parser_suspicion():
    s = score_sample(_row(), _gen(content="[answer]\n1: overruled\n2: overruled\n"
                                          "3: overruled\n4: overruled\n[/answer]"))
    assert s["no_answer_region"] is False
    if s["score"] == 0.0:
        assert s["zero_score_with_valid_region"] is True


def test_api_error_is_not_counted_as_a_wrong_answer():
    s = score_sample(_row(), _gen(error="HTTP 500: boom"))
    assert s["score"] == 0.0 and s["api_error"] is True
    assert s["no_answer_region"] is False   # no output means no format verdict


def test_truncation_is_recorded_separately():
    s = score_sample(_row(), _gen(content="thinking...", finish_reason="length"))
    assert s["truncated"] is True


def test_aggregate_groups_and_keeps_flags_out_of_the_mean():
    scored = [
        {"sample_id": "a", "task": "t1", "mode": "symbolic", "level": 1, "idx": 0,
         "score": 1.0, "api_error": False, "truncated": False,
         "no_answer_region": False, "zero_score_with_valid_region": False,
         "completion_tokens": 10},
        {"sample_id": "b", "task": "t1", "mode": "content", "level": 1, "idx": 0,
         "score": 0.0, "api_error": False, "truncated": True,
         "no_answer_region": True, "zero_score_with_valid_region": False,
         "completion_tokens": 20},
    ]
    m = aggregate(scored)
    assert m["overall"]["truncated_rate"] == 0.5
    assert m["by_mode"]["symbolic"]["mean_score"] == 1.0
    assert m["by_mode"]["content"]["mean_score"] == 0.0
    assert m["by_level"]["L01"]["n"] == 2
    assert m["by_task_mode"]["t1|content"]["n"] == 1


def test_agg_of_empty_is_empty():
    assert _agg([]) == {}


def _rows(*specs):
    """(mode, score) pairs -> scored-sample dicts."""
    return [{"sample_id": f"t__{m}__L01__{i:03d}", "task": "status_query", "mode": m,
             "level": 1, "idx": i, "score": s, "api_error": False, "truncated": False,
             "no_answer_region": False, "zero_score_with_valid_region": False,
             "completion_tokens": 10}
            for i, (m, s) in enumerate(specs)]


def test_cross_mode_groups_carry_no_score():
    """Content and symbolic are different representations of the same tasks, so a
    mean over both is not a quantity. Groups that span them must not publish one --
    a stored number gets quoted, however it is captioned."""
    from evals.scoring import aggregate

    m = aggregate(_rows(("content", 0.0), ("content", 0.0),
                        ("symbolic", 1.0), ("symbolic", 1.0)))
    for group, key in (("overall", None), ("by_task", "status_query"),
                       ("by_level", "L01")):
        node = m[group] if key is None else m[group][key]
        assert "mean_score" not in node, f"{group} published a cross-mode mean"
        assert "perfect_rate" not in node and "zero_rate" not in node
        assert node["n"] == 4                      # counts still reported
        assert "truncated_rate" in node            # mode-agnostic plumbing kept


def test_mode_split_groups_do_carry_scores():
    """Every cross-mode group has a per-mode counterpart, so refusing to average
    loses no information."""
    from evals.scoring import aggregate

    m = aggregate(_rows(("content", 0.0), ("content", 0.5),
                        ("symbolic", 1.0), ("symbolic", 1.0)))
    assert m["by_mode"]["content"]["mean_score"] == 0.25
    assert m["by_mode"]["symbolic"]["mean_score"] == 1.0
    assert m["by_level_mode"]["L01|content"]["mean_score"] == 0.25
    assert m["by_task_mode"]["status_query|symbolic"]["mean_score"] == 1.0
