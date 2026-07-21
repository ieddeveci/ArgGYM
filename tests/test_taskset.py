import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import score_answer
from evals.taskset import build_rows, cell_seed, sample_id, taskset_hash

TASKS = ["status_query", "attack"]
MODES = ["symbolic", "content"]
LEVELS = [1, 5]
SEED = 20260721


def test_sample_id_format_is_sortable_and_padded():
    assert sample_id("status_query", "content", 5, 3) == "status_query__content__L05__003"
    assert sample_id("attack", "symbolic", 15, 120) == "attack__symbolic__L15__120"


def test_cell_seed_is_stable_and_distinct_per_cell():
    a = cell_seed(SEED, "attack", "symbolic", 5)
    assert a == cell_seed(SEED, "attack", "symbolic", 5)
    assert a != cell_seed(SEED, "attack", "content", 5)
    assert a != cell_seed(SEED, "attack", "symbolic", 3)
    assert a != cell_seed(SEED + 1, "attack", "symbolic", 5)


def test_build_is_deterministic():
    a = build_rows(TASKS, MODES, LEVELS, 2, SEED)
    b = build_rows(TASKS, MODES, LEVELS, 2, SEED)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]


def test_raising_n_extends_rather_than_reshuffles():
    """Growing the taskset later must keep already-evaluated items identical."""
    small = build_rows(TASKS, MODES, LEVELS, 2, SEED)
    big = build_rows(TASKS, MODES, LEVELS, 4, SEED)
    big_by_id = {r["sample_id"]: r for r in big}
    for r in small:
        assert big_by_id[r["sample_id"]]["prompt"] == r["prompt"]


def test_gold_answers_score_one():
    """Every item's own gold answer must score 1.0, else the parser is broken
    and no model score from this taskset would mean anything."""
    for r in build_rows(TASKS, MODES, LEVELS, 2, SEED, validate_gold=False):
        assert score_answer(r["entry"]["answer"], r["entry"]) == 1.0, r["sample_id"]


def test_taskset_hash_tracks_prompts_and_kb():
    rows = build_rows(TASKS, MODES, LEVELS, 2, SEED)
    h = taskset_hash(rows, "kbsha")
    assert h == taskset_hash(rows, "kbsha")
    assert h != taskset_hash(rows, "different-kb")
    mutated = [dict(r) for r in rows]
    mutated[0]["prompt"] += " "
    assert h != taskset_hash(mutated, "kbsha")


def test_row_count_matches_grid():
    rows = build_rows(TASKS, MODES, LEVELS, 3, SEED)
    assert len(rows) == len(TASKS) * len(MODES) * len(LEVELS) * 3
