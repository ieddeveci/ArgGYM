"""The curriculum draws cells by p(1 - p), learns from every step, and agrees across ranks.

Every rank holds its own sampler, so agreement is not shared state but the same
seed and the same updates in the same order; the last test drives two of them
through accelerate's real sharding and one-batch lookahead.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rl import config as rl_config
from rl.curriculum import Curriculum, base_pass_counts, group_outcomes

ROOT = Path(__file__).resolve().parents[2]
PRIOR = {("easy", 1): (8, 8), ("mid", 1): (4, 8), ("hard", 1): (0, 8)}


def make(seed=0, prior=PRIOR, **kw):
    cells = {c: list(range(i * 100, i * 100 + 100)) for i, c in enumerate(sorted(prior))}
    return Curriculum(cells, prior, prompts_per_step=8, num_generations=8,
                      steps_per_generation=32, seed=seed, **kw)


def outcomes_for(rows, cur, successes):
    """One group per drawn row, `successes` of 8 each."""
    cell_of = {i: c for c, idx in cur.cells.items() for i in idx}
    return [(cell_of[r], successes, 8) for r in rows]


def test_mid_difficulty_cells_weigh_most_and_no_cell_weighs_nothing():
    w = make().weights()
    assert w[("mid", 1)] > w[("easy", 1)] > 0
    assert w[("easy", 1)] == pytest.approx(w[("hard", 1)])
    # 0/8 and 8/8 are not certainties under the Beta(1, 1) prior.
    assert make().pass_rate(("hard", 1)) == pytest.approx(1 / 10)


def test_rows_come_without_replacement_within_a_cell():
    cur = make(prior={("only", 1): (4, 8)})
    rows = [r for _ in range(25) for r in cur.draw()]
    assert sorted(rows[:100]) == list(range(100))


def test_the_same_seed_and_outcomes_give_the_same_draws():
    a, b = make(seed=3), make(seed=3)
    for step in range(6):
        ra, rb = a.draw(), b.draw()
        assert ra == rb
        a.update(outcomes_for(ra, a, step % 3))
        b.update(outcomes_for(rb, b, step % 3))
    assert a.state_dict() == b.state_dict()


def test_outcomes_move_the_draw_toward_cells_with_signal():
    cur = make(prior={("a", 1): (4, 8), ("b", 1): (4, 8)})
    for _ in range(20):
        cur.update([(("a", 1), 8, 8)])  # cell a saturates
    w = cur.weights()
    assert w[("b", 1)] > 5 * w[("a", 1)]


def test_old_evidence_halves_after_ten_observations_of_its_cell():
    cur = make(prior={("a", 1): (8, 8)})
    for _ in range(10):
        cur.update([(("a", 1), 0, 8)])
    s, f = cur.counts[("a", 1)]
    assert s == pytest.approx(4.0)


def test_a_truncated_completion_counts_as_a_failure():
    recs = [{"id": "x", "task": "t", "level": 2, "success": True, "truncated": t}
            for t in (False, False, True)]
    recs.append({"id": "x", "task": "t", "level": 2, "success": None, "truncated": False})
    assert group_outcomes(recs) == [(("t", 2), 2, 3)]


def test_the_base_eval_supplies_every_cell_the_config_trains():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b-curriculum.yaml"))
    prior = base_pass_counts(str(ROOT / cfg["curriculum"]["prior"]))
    import arggym

    cells = {(t, lv) for t in arggym.task_names() for lv in cfg["data"]["levels"]}
    assert cells <= set(prior) and all(n == 8 for _, n in prior.values())


def loaders(curs):
    """Each rank's dataloader, as TRL builds it and accelerate shards it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from accelerate.data_loader import prepare_data_loader

    return [iter(prepare_data_loader(
        torch.utils.data.DataLoader(range(300), batch_size=32, sampler=cur),
        num_processes=len(curs), process_index=r, put_on_device=False))
        for r, cur in enumerate(curs)]


def train_step(iters, curs, step):
    """One optimizer step as Transformers runs it: fetch the whole accumulation
    window, then generate and score, then every rank adds the gathered outcomes."""
    windows = [[next(it).tolist() for _ in range(32)] for it in iters]
    for w in windows:
        # A rank's completions are whole groups of 8, the same block every micro-step.
        assert all(b == w[0] for b in w)
        assert all(w[0].count(i) == 8 for i in set(w[0]))
    rows = sorted({i for w in windows for i in w[0]})
    for cur in curs:
        cur.update(outcomes_for(rows, cur, step * 3 % 9))
    return rows


def test_every_rank_draws_the_same_prompts_and_learns_with_a_lag_of_two_steps():
    ranks = [make(seed=5) for _ in range(2)]
    iters = loaders(ranks)
    for step in range(4):
        rows = train_step(iters, ranks, step)
        assert len(rows) == 8 and rows == sorted(ranks[0].draws[step]["rows"])
        # Block step+1 was drawn before step's outcomes arrived.
        assert ranks[0]._before[step + 1]["observed"] == step
    assert ranks[0].draws == ranks[1].draws


def test_a_resumed_sampler_draws_what_the_original_would_have(tmp_path):
    run = [make(seed=1) for _ in range(2)]
    iters = loaders(run)
    for step in range(2):
        train_step(iters, run, step)
    run[0].save(tmp_path, global_step=2)  # rank 0 saves, mid-lookahead
    later = [train_step(iters, run, step) for step in (2, 3)]

    resumed = [make(seed=1) for _ in range(2)]
    for cur in resumed:
        cur.load(tmp_path)
    iters = loaders(resumed)
    assert [train_step(iters, resumed, step) for step in (2, 3)] == later
