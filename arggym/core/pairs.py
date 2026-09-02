"""Scoring for tasks that answer one `key: status` line per claim.

status_query, semantics_query and perturbation each turn an answer into a label map and score
it by F1 over keys. A key given two different statuses is a hedge: it counts as one prediction
that is never a true positive, so it costs the same as a single wrong answer in either line
order. Identical repeats of a line are not a hedge.
"""
from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, NamedTuple, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)


def collect(matches: Iterable[Tuple[K, str]]) -> Dict[K, List[str]]:
    """Distinct statuses per key, in the order seen."""
    pred: Dict[K, List[str]] = {}
    for key, status in matches:
        stats = pred.setdefault(key, [])
        if status not in stats:
            stats.append(status)
    return pred


class PairF1(NamedTuple):
    tp: int
    precision: float
    recall: float
    f1: float
    exact_match: bool
    contradicted: list


def pair_f1(pred: Dict[K, List[str]], gold: Dict[K, str]) -> PairF1:
    """F1 over keys. A key with more than one status never matches gold; each key counts once."""
    tp = sum(1 for k, s in pred.items() if s == [gold.get(k)])
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    contradicted = sorted(k for k, s in pred.items() if len(s) > 1)
    return PairF1(tp, precision, recall, f1, tp == len(gold) == len(pred), contradicted)
