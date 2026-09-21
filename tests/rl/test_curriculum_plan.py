from __future__ import annotations

import arggym

from rl.generate import _stage_quotas, _verify_stage_plan

BANDS = {
    "easy": [1, 2, 3, 4, 5],
    "medium": [6, 7, 8, 9, 10],
    "hard": [11, 12, 13, 14, 15],
}
EXPECTED = (
    {"easy": 720, "medium": 504, "hard": 216},
    {"easy": 576, "medium": 432, "hard": 432},
    {"easy": 288, "medium": 432, "hard": 720},
)


def test_article_curriculum_is_exact_and_factor_balanced():
    spec = arggym.load_spec("tasksets/standard.yaml")
    for stage_index, expected in enumerate(EXPECTED):
        q = _stage_quotas(spec, stage_index, BANDS)
        _verify_stage_plan(spec, q, expected, BANDS)
        assert sum(q.values()) == 1440
        assert {sum(v for (tt, _l, _o), v in q.items() if tt == t) for t in spec.tasks} == {120}
        assert {sum(v for (_t, _l, oo), v in q.items() if oo == o) for o in spec.orderings} == {360}
        assert {
            sum(v for (tt, _l, oo), v in q.items() if tt == t and oo == o)
            for t in spec.tasks
            for o in spec.orderings
        } == {30}


def test_smoke_plan_covers_all_tasks_bands_and_orderings():
    from rl.generate import _smoke_quotas

    spec = arggym.load_spec("tasksets/standard.yaml")
    train = _smoke_quotas(spec, dev=False)
    dev = _smoke_quotas(spec, dev=True)
    assert sum(train.values()) == 24
    assert sum(dev.values()) == 12
    assert {t for (t, _l, _o) in train} == set(spec.tasks)
    assert {t for (t, _l, _o) in dev} == set(spec.tasks)
    assert {o for (_t, _l, o) in train} == set(spec.orderings)
    assert {o for (_t, _l, o) in dev} == set(spec.orderings)
    assert {"easy", "medium", "hard"} == {
        "easy" if l <= 5 else "medium" if l <= 10 else "hard" for (_t, l, _o) in train
    }
