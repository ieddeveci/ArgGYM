"""A pass-rate curriculum: each step's prompts come from the (task, level) cells the
policy currently solves about half the time.

A GRPO group whose completions all succeed or all fail has zero advantage, and on
Qwen3.5-4B the level number says little about which groups those will be: base
success is about 1.0 on claim_chain L1-3, 0.0 on perturbation L2-4, and flat on
preference_construction across all fifteen levels. So the sampler keeps a pass
rate per cell and draws cells in proportion to p(1 - p), the variance of one
completion's success and so the expected gradient signal of its group.

- The pass rate starts from the base model's eval (8 samples a cell) and is read
  through a Beta(1, 1) prior, so 0/8 and 8/8 are not certainties.
- Every group a step generates adds its successes out of its completions to its
  cell, a truncated completion counting as a failure, as it does in eval. Before
  each addition the cell's counts are multiplied by 2^(-1/half_life): an
  observation weighs half as much after `half_life` later groups of the same cell.
- Weights get a `floor`, so a saturated or impossible cell is still drawn now and
  then and its estimate can move.
- A step's prompts are `prompts_per_step` cell draws with replacement; rows within
  a cell come from a shuffled list without replacement, reshuffled when used up.

The sampler is TRL's `RepeatSampler` order with the block chosen here: each
prompt `num_generations` times in a row, the whole block repeated
`steps_per_generation` times, one block per optimizer step. Every rank holds its
own copy, seeded alike and updated from the gathered rewards of the whole batch,
so every rank draws the same block; accelerate then hands each rank its slice.

Lag: two steps. Transformers fetches a step's whole accumulation window before
running it, and accelerate reads one batch beyond that, so block t+1 is drawn at
the start of step t, before step t's rewards exist. An outcome at step t first
shapes block t+2. (Measured in the smoke run: the state saved at step 2 had seen
one step's outcomes when it drew block 2.)

Resuming: a checkpoint at step G stores the state from just before block G was
drawn, plus the updates that arrived after that draw. A resumed sampler draws
block G again and applies those updates right after it, as the original did.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[str, int]
STATE_FILE = "curriculum.json"


def base_pass_counts(metrics_path: str) -> Dict[Cell, Tuple[int, int]]:
    """(successes, scored) per (task, level) from an eval run's `metrics.json`."""
    by_cell = json.loads(Path(metrics_path).read_text())["by_task_level"]
    out = {}
    for key, m in by_cell.items():
        task, level = key.split("|L")
        n = m["n_scored"]
        out[(task, int(level))] = (round(m["success_rate"] * n), n)
    return out


def group_outcomes(records: Sequence[Dict[str, Any]]) -> List[Tuple[Cell, int, int]]:
    """(cell, successes, completions) per group, a truncated completion counting as a failure.

    A completion the scorer refused says nothing about the cell and is left out.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["success"] is not None:
            groups[r["id"]].append(r)
    return [((g[0]["task"], g[0]["level"]),
             sum(bool(r["success"]) and not r["truncated"] for r in g), len(g))
            for g in groups.values()]


class Curriculum:
    """Iterable of dataset indices, for TRL's train dataloader. See the module docstring."""

    def __init__(self, cells: Dict[Cell, List[int]], prior: Dict[Cell, Tuple[int, int]],
                 prompts_per_step: int, num_generations: int, steps_per_generation: int,
                 seed: int, half_life: float = 10.0, floor: float = 0.01) -> None:
        self.cells = {c: list(rows) for c, rows in sorted(cells.items())}
        self.prompts_per_step = prompts_per_step
        self.num_generations = num_generations
        self.steps_per_generation = steps_per_generation
        self.decay = 0.5 ** (1.0 / half_life)
        self.floor = floor
        self.rng = np.random.Generator(np.random.PCG64(seed))
        # Successes and failures, decayed. A cell the base eval never ran starts empty.
        self.counts = {c: list(map(float, (prior[c][0], prior[c][1] - prior[c][0])))
                       if c in prior else [0.0, 0.0] for c in self.cells}
        self.order: Dict[Cell, List[int]] = {}
        self.pos: Dict[Cell, int] = {}
        for c in self.cells:
            self._reshuffle(c)
        self.blocks = 0  # blocks drawn
        self.observed = 0  # blocks whose outcomes have been added
        self.draws: Dict[int, Dict[str, Any]] = {}
        self._before: Dict[int, Dict[str, Any]] = {}  # state just before drawing block k
        self._updates: List[Tuple[int, list]] = []  # (blocks drawn at the time, outcomes)
        self._pending: List[Tuple[int, list]] = []  # updates a resumed run replays

    # ------------------------------------------------------------ estimates

    def pass_rate(self, cell: Cell) -> float:
        s, f = self.counts[cell]
        return (s + 1.0) / (s + f + 2.0)

    def weights(self) -> Dict[Cell, float]:
        return {c: self.pass_rate(c) * (1 - self.pass_rate(c)) + self.floor for c in self.cells}

    # ------------------------------------------------------------ drawing

    def _reshuffle(self, cell: Cell) -> None:
        self.order[cell] = [int(i) for i in self.rng.permutation(self.cells[cell])]
        self.pos[cell] = 0

    def draw(self) -> List[int]:
        """The next block's prompts, as dataset indices."""
        self._before[self.blocks] = self.state_dict()
        for old in [b for b in self._before if b < self.blocks - 2]:
            del self._before[old]
        cells = list(self.cells)
        w = np.array([self.weights()[c] for c in cells])
        picks = self.rng.choice(len(cells), size=self.prompts_per_step, p=w / w.sum())
        rows, drawn = [], []
        for i in picks:
            c = cells[int(i)]
            if self.pos[c] == len(self.order[c]):
                self._reshuffle(c)
            rows.append(self.order[c][self.pos[c]])
            self.pos[c] += 1
            p = self.pass_rate(c)
            drawn.append({"task": c[0], "level": c[1], "pass_rate": p, "signal": p * (1 - p)})
        self.draws[self.blocks] = {"block": self.blocks, "cells": drawn, "rows": rows}
        self.blocks += 1
        while self._pending and self._pending[0][0] == self.blocks:
            self.update([((t, lv), s, n) for t, lv, s, n in self._pending.pop(0)[1]])
        return rows

    def __iter__(self) -> Iterator[int]:
        while True:
            block = [i for i in self.draw() for _ in range(self.num_generations)]
            for _ in range(self.steps_per_generation):
                yield from block

    def __len__(self) -> int:
        # Never exhausted; the trainer stops at `max_steps`.
        return 10 ** 9

    # ------------------------------------------------------------ learning

    def update(self, outcomes: Iterable[Tuple[Cell, int, int]]) -> None:
        """Add one block's groups. Called once per optimizer step, on every rank alike."""
        record = self.draws.get(self.observed)
        added = []
        for cell, successes, n in outcomes:
            if cell not in self.counts:
                continue
            s, f = self.counts[cell]
            self.counts[cell] = [s * self.decay + successes, f * self.decay + n - successes]
            added.append({"task": cell[0], "level": cell[1], "successes": successes, "n": n})
        if record is not None:
            record["outcomes"] = added
        self.observed += 1
        self._updates.append((self.blocks, [[a["task"], a["level"], a["successes"], a["n"]]
                                            for a in added]))
        oldest = min(self._before, default=self.blocks)
        self._updates = [u for u in self._updates if u[0] > oldest]

    # ------------------------------------------------------------ resuming

    def state_dict(self) -> Dict[str, Any]:
        return {"counts": [[c[0], c[1], s, f] for c, (s, f) in self.counts.items()],
                "order": [[c[0], c[1], o, self.pos[c]] for c, o in self.order.items()],
                "rng": self.rng.bit_generator.state, "blocks": self.blocks,
                "observed": self.observed}

    def state_for_step(self, global_step: int) -> Dict[str, Any]:
        """What a run resumed at `global_step` needs to continue as this one does.

        A checkpoint at step G has finished blocks 0..G-1, but the dataloader has
        usually drawn block G already, and an update may have arrived since. So:
        the state from just before block G was drawn, and the updates after it.
        """
        if global_step in self._before:
            return {"state": copy.deepcopy(self._before[global_step]),
                    "after": [u for u in self._updates if u[0] > global_step]}
        if global_step == self.blocks:
            return {"state": self.state_dict(), "after": []}
        raise RuntimeError(f"no curriculum state for step {global_step} "
                           f"(blocks drawn: {self.blocks})")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.counts = {(t, lv): [s, f] for t, lv, s, f in state["counts"]}
        self.order = {(t, lv): list(o) for t, lv, o, _ in state["order"]}
        self.pos = {(t, lv): p for t, lv, _, p in state["order"]}
        self.rng.bit_generator.state = state["rng"]
        self.blocks, self.observed = state["blocks"], state["observed"]
        self.draws, self._before, self._updates, self._pending = {}, {}, [], []

    def save(self, directory: Path, global_step: int) -> None:
        (Path(directory) / STATE_FILE).write_text(json.dumps(self.state_for_step(global_step)))

    def load(self, directory: Path) -> None:
        saved = json.loads((Path(directory) / STATE_FILE).read_text())
        self.load_state_dict(saved["state"])
        self._pending = [(b, outs) for b, outs in saved["after"]]

    # ------------------------------------------------------------ reporting

    def table(self) -> List[Dict[str, Any]]:
        """Every cell's estimate, for a heatmap."""
        w = self.weights()
        total = sum(w.values())
        return [{"task": c[0], "level": c[1], "pass_rate": self.pass_rate(c),
                 "signal": self.pass_rate(c) * (1 - self.pass_rate(c)),
                 "draw_share": w[c] / total, "evidence": sum(self.counts[c])}
                for c in self.cells]


def summary(draw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Scalars for one step's draw."""
    if not draw:
        return {}
    cells = draw["cells"]
    return {"curriculum/mean_signal": sum(c["signal"] for c in cells) / len(cells),
            "curriculum/mean_pass_rate": sum(c["pass_rate"] for c in cells) / len(cells),
            "curriculum/distinct_cells": float(len({(c["task"], c["level"]) for c in cells}))}
