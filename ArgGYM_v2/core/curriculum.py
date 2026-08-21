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
            # C8 (junction) is deliberately NOT in the attack mix. It generates and verifies
            # correctly for defence and attack_defense, but attack items built on it are rejected
            # downstream and the cause was not isolated. Left out rather than shipped broken.
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


JUNCTION_START = 5
JUNCTION_SHARE = 0.20


def junctions_for(level: int, n_rules: int, start: int = JUNCTION_START,
                  share: float = JUNCTION_SHARE, solve: bool = True) -> int:
    """How many junctions a theory of `n_rules` rules should carry.

    A CONSTANT SHARE from level 5 upward, not a ramp. The earlier `junction_budget` scaled with LEVEL
    against a per-task cap, which meant density tracked whatever ratio the cap happened to bear to the
    task's rule count -- 22% in formalization, 7% in claim_chain, from the same machinery.

    Sizing against rule count instead makes 20% mean the same thing everywhere, and makes it hold at
    every level rather than only at the top of the curriculum.

    STRICT rules are excluded from the count by the caller: a junction on a strict step contributes no
    new cut point, since a strict rule cannot be undercut, so the branch would be decoration.
    """
    if level < start:
        return 0
    # Solve for the FIXED POINT. Each junction adds its own branch rules -- one or two single-antecedent
    # rules -- so placing j junctions in a theory of n base rules gives n + j*extra rules, of which j
    # are multi-antecedent. Sizing against n alone undershoots: measured 14-17% where 20% was asked.
    #
    #     j / (n + j*extra) = share   =>   j = share*n / (1 - share*extra)
    #
    # with extra ~1.5 (a mix of binary and ternary junctions).
    # `solve=False` for callers that ALREADY model the rules a junction adds. Passing an estimate of
    # the base count and then solving the fixed point on top of it double-corrects: measured, the two
    # tasks whose filler top-up models added rules explicitly landed at 27-31% where 20% was asked.
    if not solve:
        return max(1, round(share * max(1, n_rules)))
    extra = 1.0
    denom = max(0.05, 1.0 - share * extra)
    return max(1, round(share * max(1, n_rules) / denom))


def junction_budget(level: int, cap: int, start: int = JUNCTION_START) -> int:
    """How many junctions an item at `level` should carry, rising linearly to `cap` at level 15.

    Junctions begin at level 6 rather than 8. Below that is the easy band, where branch structure would
    flatten the gradient the curriculum depends on.

    `cap` is PER TASK and is not a style choice -- it is set by the task's natural repeating unit and by
    where the junction lands:

      * a junction adds TWO directives (a branch root and a branch rule), so a task with a small theory
        saturates quickly. `formalization` runs at 18% junction density at level 15 while `claim_chain`
        runs at 0.8%, and the same absolute count would mean very different things.
      * where a junction enters the ANSWER rather than only the theory, each one adds two directives to
        what the model must produce. `claim_chain` traces its junctions, so its cap is set by how long
        an answer stays reasonable, not by how many the theory could hold.
      * `defeat_diagnosis` is capped at its route count, which is fixed at three above the easy band for
        an unrelated reason: the kind-guessing floor is 1/n_routes.
    """
    if level < start:
        return 0
    span = max(1, 15 - start)
    return max(1, min(cap, round(1 + (level - start) * (cap - 1) / span)))


JUNCTION_CAPS = {
    # Raised across the board. Measured density before this was 1-8% of rules, and three tasks sat at
    # exactly ONE junction at every level from 8 to 15 -- claim_chain 1/129, defeat_diagnosis 1/75,
    # the attack_defense family 1/46. A structure present once in a hundred rules is a curiosity, not
    # a tested mechanism.
    # Raised again. Multi-premise rules are the NORMAL case in argumentation -- argument schemes are
    # mostly two or three premises -- so a few percent of rules is unrepresentative. These caps target
    # roughly 10-20% of rules being multi-antecedent at level 15.
    "status_query": 20,
    "preference_construction": 14,
    "perturbation": 12,
    # Formalization theories are the SMALLEST in the suite (19 rules at L15), so a cap of 10
    # put junction density at 53% -- multi-antecedent became the majority case rather than a
    # substantial minority. Sized against its own rule count, not by copying another task.
    "formalization": 5,
    "claim_chain": 10,               # traced, so each adds two or three directives to the answer
    "defeat_diagnosis": 9,           # up to three per route across three routes
    "counter_argument": 8,
    "attack_defense": 10,
    "semantics_query": 2,   # theories capped at 26 directives, so the budget must stay small
}

# Share of junctions that take a THIRD antecedent, from this level upward. Two antecedents test that
# either branch is a cut point; three tests that a model tracks several simultaneous dependencies, and
# is the arity argument schemes actually use -- expert opinion is a three-premise scheme.
NEGATED_BRANCH_SHARE = 0.5


def negated_branch(index: int, share: float = NEGATED_BRANCH_SHARE) -> bool:
    """Whether junction branch `index` should be built from a NEGATED literal.

    Junction branches were positive-only, and since junctions now make up a fifth of all rules they
    diluted negation badly as theories grew: claim_chain fell from 15% negated directives at level 4 to
    5% at level 15, perturbation from 17% to 6%. The negation gadgets are fixed in number, so growing
    everything around them lowers the share.

    A negated branch is not decoration. `-x => y` is the `negated_antecedent` role -- a rule firing
    from the absence of something -- and it puts the branch's own contested pair in play, so cutting
    that branch means reasoning about which side of the pair won.
    """
    step = max(1, round(1 / max(share, 0.01)))
    return (index % step) == 0


TERNARY_FROM_LEVEL = 9
TERNARY_SHARE = 0.4


def wants_ternary(level: int, index: int) -> bool:
    """Whether junction number `index` at this level should take a third antecedent."""
    if level < TERNARY_FROM_LEVEL:
        return False
    return (index % max(1, round(1 / TERNARY_SHARE))) == 0


@dataclass(frozen=True)
class LanguageProfile:
    """A restricted ASPIC+ language fragment.

    Every task previously used all six directive kinds in every item, so nothing tested whether a
    model's competence is uniform across fragments or leans on one kind being present. Each fragment
    removes different machinery, and the removals are not degrees of difficulty -- they are different
    problems:

      P_D   premises + defeasible. No strict rules, so nothing is unattackable; no axioms, so every
            root can be undermined. All three attack forms available. The purely defeasible core.

      P_S   premises + strict. NO DEFEASIBLE RULES, so measured: rebut is unavailable (two strict
            rules with contrary conclusions leave BOTH justified without transposition, and with
            transposition both undecided) and undercut is unavailable (an undercutter must itself be a
            defeasible rule). ONLY UNDERMINE WORKS, and only a premise preference resolves anything.
            This is the monotonic core with uncertainty confined to the leaves -- what the literature
            calls the classical part, and a genuinely different reasoning problem rather than an
            easier one.

      P_S_D premises + strict + defeasible, no axioms. Everything is attackable somewhere; there is no
            unassailable ground.

      FULL  all six kinds. The current suite.

    A task declares which fragments it supports. Several cannot support all four: `attack_defense`
    requires a strict final rule so it cannot run P_D, and `preference_construction` needs conflicts a
    preference can settle so it cannot run P_S.
    """
    name: str
    kinds: FrozenSet[str]
    note: str = ""

    def permits(self, kind: str) -> bool:
        return kind in self.kinds

    def filter(self, ops):
        return [o for o in ops if o.kind in self.kinds]


_PREF = {"prefer_rule", "prefer_premise"}

PROFILES: Dict[str, LanguageProfile] = {
    "P_D": LanguageProfile(
        "P_D", frozenset({"premise", "defeasible"} | _PREF),
        "premises and defeasible rules; all three attack forms available"),
    "P_S": LanguageProfile(
        "P_S", frozenset({"premise", "strict", "prefer_premise"}),
        "premises and strict rules; ONLY undermine works, and only premise preferences resolve"),
    "P_S_D": LanguageProfile(
        "P_S_D", frozenset({"premise", "strict", "defeasible"} | _PREF),
        "no axioms, so nothing is unassailable at the root"),
    "FULL": LanguageProfile(
        "FULL", frozenset({"premise", "axiom", "defeasible", "strict"} | _PREF),
        "the full language"),
}

# Which fragments each task can express. Measured, not assumed -- a task listed here must generate
# and score correctly under every fragment it claims.
TASK_PROFILES: Dict[str, Tuple[str, ...]] = {
    "status_query": ("P_D", "P_S", "P_S_D", "FULL"),
    # P_D ONLY, and by construction rather than by restriction. Measured: the task emits premises
    # and defeasible rules exclusively. Its divergence structures -- floating conclusions, odd cycles,
    # reinstatement -- are all defeasible; a strict rule makes that part of the theory monotonic,
    # which cannot create divergence between semantics and can destroy it. Declaring P_S_D or FULL
    # here would be a label with nothing behind it.
    "semantics_query": ("P_D",),
    "claim_chain": ("P_D", "P_S_D", "FULL"),
    "defeat_diagnosis": ("P_D", "P_S_D", "FULL"),
    "perturbation": ("P_D", "P_S_D", "FULL"),
    "preference_construction": ("P_D", "P_S_D", "FULL"),
    "counter_argument": ("P_S_D", "FULL"),
    "formalization": ("P_D", "P_S_D", "FULL"),
    # P_S_D reached by dropping C3 and C4, the only axiom-root configurations -- measured, and five
    # configs remain at level 10+, above the want_distinct floor of three. P_D is NOT reachable: the
    # task requires at least one strict-final chain, which is what closes the rebut route on that chain
    # and forces per-chain discrimination.
    "attack_defense": ("P_S_D", "FULL"),
}
