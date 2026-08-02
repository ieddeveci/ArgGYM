from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ATTACK, DEFENCE, MIXED = "attack", "defence", "attack_defense"
GOAL_UNDECIDED, GOAL_OVERRULED, GOAL_JUSTIFIED = "UNDECIDED", "OVERRULED", "JUSTIFIED"
LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"


COST_TABLE: Dict[Tuple[str, int], float] = {
    ("C1", 2): 1.5, ("C1", 3): 1.6, ("C1", 4): 1.7, ("C1", 6): 1.8, ("C1", 9): 1.8,
    ("C2", 2): 4.0, ("C2", 3): 4.0, ("C2", 4): 4.0, ("C2", 6): 4.0, ("C2", 9): 4.0,
    ("C3", 2): 2.7, ("C3", 3): 2.8, ("C3", 4): 2.8, ("C3", 6): 2.9, ("C3", 9): 2.9,
    ("C4", 2): 6.0, ("C4", 3): 5.5, ("C4", 4): 5.3, ("C4", 6): 5.2, ("C4", 9): 5.1,
    ("C5", 2): 2.5, ("C5", 3): 2.6, ("C5", 4): 2.7, ("C5", 6): 2.8, ("C5", 9): 2.8,
    ("C6", 2): 4.0, ("C6", 3): 5.5, ("C6", 4): 7.0, ("C6", 6): 10.0, ("C6", 9): 14.5,
    ("C7", 2): 1.5, ("C7", 3): 4.0, ("C7", 4): 5.3, ("C7", 6): 8.0, ("C7", 9): 12.0,
}


def chain_cost(config: str, depth: int) -> float:
    keys = sorted({d for (c, d) in COST_TABLE if c == config})
    return COST_TABLE[(config, min(keys, key=lambda k: abs(k - depth)))]


RELEVANCE_WEIGHT = 0.8
NOISE_WEIGHT = 0.6


def relevance_cost(chain_depths: Sequence[int], target_depth: int,
                   noise_rules: int) -> float:
    dead_in_chains = sum(max(0, d - target_depth) for d in chain_depths)
    return round(RELEVANCE_WEIGHT * dead_in_chains + NOISE_WEIGHT * noise_rules, 2)


SURVIVAL_BONUS = 2.0
INTERACTION_BONUS = 3.0


@dataclass
class Goal:
    claim: str
    mode: str
    want: str
    target_depth: int = 0
    chain_configs: Tuple[str, ...] = ()


@dataclass
class ItemSpec:
    level: int
    ordering: str
    mode: str
    n_attack_goals: int
    n_defence_goals: int
    min_required_moves: int
    min_rejected_moves: int
    n_chains_min: int
    n_chains_max: int
    depth_min: int
    depth_max: int
    target_depth_min: int
    target_depth_max: int
    configs: Tuple[str, ...]
    disjoint: bool
    require_distinct_configs: int
    n_noise_components: int
    require_survival: bool
    require_interaction: bool
    require_minimal: bool

    def satisfied_by(self, required: Optional[int], rejected: Optional[int]) -> bool:
        if required is None or rejected is None:
            return False
        return required >= self.min_required_moves and rejected >= self.min_rejected_moves


def _configs_for(L: int) -> Tuple[str, ...]:
    return (("C1",) if L <= 2 else
            ("C1", "C2") if L <= 4 else
            ("C1", "C2", "C3") if L <= 6 else
            ("C1", "C2", "C3", "C7") if L <= 8 else
            ("C1", "C2", "C3", "C4", "C7") if L <= 10 else
            ("C1", "C2", "C3", "C4", "C5", "C6", "C7"))


_MEASURED_FLOORS: Dict[int, Tuple[int, int]] = {
    3: (3, 6), 5: (3, 8), 7: (4, 8), 9: (5, 13), 15: (7, 19),
}


def _interp(L: int, idx: int) -> int:
    ks = sorted(_MEASURED_FLOORS)
    if L <= ks[0]:
        return _MEASURED_FLOORS[ks[0]][idx]
    if L >= ks[-1]:
        return _MEASURED_FLOORS[ks[-1]][idx]
    lo = max(k for k in ks if k <= L)
    hi = min(k for k in ks if k >= L)
    if lo == hi:
        return _MEASURED_FLOORS[lo][idx]
    a, b = _MEASURED_FLOORS[lo][idx], _MEASURED_FLOORS[hi][idx]
    return int(round(a + (b - a) * (L - lo) / (hi - lo)))


def _floor_required(L: int) -> int:
    if L in _MEASURED_FLOORS:
        return _MEASURED_FLOORS[L][0]
    return max(2, _interp(L, 0))


def _floor_rejected(L: int) -> int:
    if L in _MEASURED_FLOORS:
        return _MEASURED_FLOORS[L][1]
    return max(2, _interp(L, 1))


def _mode_for(L: int, seed_parity: int) -> Tuple[str, int, int]:
    if L <= 3:
        return (ATTACK, 1, 0) if seed_parity == 0 else (DEFENCE, 0, 1)
    if L <= 6:
        return [(ATTACK, 1, 0), (DEFENCE, 0, 1), (MIXED, 1, 1)][seed_parity % 3]
    if L <= 10:
        return [(MIXED, 1, 1), (MIXED, 2, 1), (ATTACK, 2, 0), (DEFENCE, 0, 2)][seed_parity % 4]
    return [(MIXED, 2, 1), (MIXED, 1, 2), (MIXED, 2, 2), (ATTACK, 3, 0),
            (MIXED, 3, 2)][seed_parity % 5]


def spec_for(level: int, ordering: str = LAST_LINK, variant: int = 0) -> ItemSpec:
    L = max(1, min(15, level))
    mode, na, nd = _mode_for(L, variant)
    n_max = min(2 + L // 2, 10)
    d_max = min(2 + L // 2, 9)
    cfgs = _configs_for(L)
    td_min = 2
    td_max = d_max if L < 5 else max(2, d_max - 2)
    noise = 0 if L < 6 else min(1 + (L - 6) // 4, 3)
    noise_rules = noise * 3

    return ItemSpec(
        level=L, ordering=ordering, mode=mode,
        n_attack_goals=na, n_defence_goals=nd,
        min_required_moves=_floor_required(L),
        min_rejected_moves=_floor_rejected(L),
        n_chains_min=2 if L >= 2 else 1, n_chains_max=n_max,
        depth_min=2, depth_max=d_max,
        target_depth_min=td_min, target_depth_max=td_max,
        configs=cfgs,
        disjoint=L >= 8,
        require_distinct_configs=1 if L <= 4 else (2 if L <= 8 else 3),
        n_noise_components=noise,
        require_survival=L >= 9,
        require_interaction=(mode == MIXED and L >= 7),
        require_minimal=True,
    )


def describe(spec: ItemSpec) -> str:
    return (f"L{spec.level} {spec.mode} ({spec.n_attack_goals}A/{spec.n_defence_goals}D) "
            f"{spec.ordering.split('_')[0]}-link | chains {spec.n_chains_min}-{spec.n_chains_max} "
            f"depth {spec.depth_min}-{spec.depth_max} target-depth "
            f"{spec.target_depth_min}-{spec.target_depth_max} | configs {list(spec.configs)} "
            f"| disjoint={spec.disjoint} noise={spec.n_noise_components} "
            f"survival={spec.require_survival} interaction={spec.require_interaction} "
            f"| require >={spec.min_required_moves} moves, >={spec.min_rejected_moves} dead ends"
            + ("" if spec.level in _MEASURED_FLOORS else " (interpolated)"))
