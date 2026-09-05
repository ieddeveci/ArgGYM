"""The inspector is a view onto the scorer, so it must not have its own answer contract.

It is excluded from ruff (it embeds JavaScript in string literals) and it duplicates the
per-task dispatch that `arggym.core.rows` does for rows, so it is the file most able to
drift without anything noticing. When the answer template changed, the question this
suite answers was asked by hand.

What is pinned is the seam, not the markup: every mode reaches a real item, the question
asks for the template the dataset defaults to, and the score the panel prints is the score
the task's own scorer returns for that text -- fenced or bare, since the fence is delivery
(`docs/dataset-contract.md` section 1).
"""
from __future__ import annotations

import pytest

from arggym.core.answers import DEFAULT_TEMPLATE
from arggym.inspector import MODES, app

LEVEL, SEED, ORDERING = 3, 1, "last_link_elitist"
MODE_KEYS = [m[0] for m in MODES]


@pytest.fixture(scope="module")
def client():
    return app.test_client()


@pytest.fixture(scope="module")
def items(client):
    built = {}
    for key in MODE_KEYS:
        d = client.post("/api/generate", json={"mode": key, "level": LEVEL, "seed": SEED,
                                               "ordering": ORDERING}).get_json()
        assert d and d.get("token"), f"{key}: {d}"
        built[key] = d
    return built


def grade(client, token, answer):
    r = client.post("/api/grade", json={"token": token, "answer": answer}).get_json()
    assert r and r.get("ok"), r
    return r["result"]


def test_every_mode_in_the_dropdown_builds_an_item(items):
    """A mode the UI offers and the API cannot serve is a dead menu entry."""
    assert sorted(items) == sorted(MODE_KEYS)


@pytest.mark.parametrize("key", MODE_KEYS)
def test_the_question_says_nothing_about_where_to_put_the_answer(key, items):
    """Delivery is the harness's, so the question states the task and stops there."""
    prompt = items[key]["prompt"]
    assert DEFAULT_TEMPLATE.instruction not in prompt
    assert DEFAULT_TEMPLATE.open not in prompt


@pytest.mark.parametrize("key", MODE_KEYS)
def test_the_reference_scores_one_in_the_panel(key, items):
    assert items[key]["ref_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("key", MODE_KEYS)
def test_a_pasted_reference_scores_the_same_bare_or_fenced(key, items, client):
    """The two shapes a user pastes into the box: the answer, or a whole completion.

    The page extracts, because it is the harness here. The dataset would read the
    fenced form as an answer whose first and last lines are unreadable.
    """
    d = items[key]
    bare = grade(client, d["token"], d["reference"])
    fenced = grade(client, d["token"],
                   f"Let me think.\n{DEFAULT_TEMPLATE.wrap(d['reference'])}")
    assert bare["score"] == pytest.approx(1.0)
    assert fenced["score"] == pytest.approx(bare["score"])
    assert fenced["success"] is bare["success"] is True


@pytest.mark.parametrize("key", MODE_KEYS)
@pytest.mark.parametrize("answer", ["", "I am not sure."])
def test_every_graded_result_carries_the_four_top_level_fields(key, answer, items, client):
    """The panel prints `success` beside the score because they differ: `formalization`
    succeeds below 1.0. A result missing the field would render `undefined`."""
    r = grade(client, items[key]["token"], answer)
    assert set(r) == {"score", "success", "reason", "diagnostics"}
    assert r["success"] is False and r["score"] == 0.0


def test_an_expired_token_is_reported_rather_than_raising(client):
    r = client.post("/api/grade", json={"token": "nope", "answer": "x"}).get_json()
    assert r["ok"] is False and "expired" in r["error"]


def test_the_page_states_the_scoring_rules_it_actually_applies(client):
    """The card claimed the construction formula for all twelve modes; six score
    another way, and `success` is each task's own bar rather than `score == 1`."""
    page = client.get("/").get_data(as_text=True)
    assert "Construction tasks: score = 0.5" in page
    assert "Query tasks score their own way" in page
    assert "this page extracts it, as an evaluation harness would." in page
